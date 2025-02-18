# --------------------------------------------------------
# References:
# multimae: https://github.com/EPFL-VILAB/MultiMAE
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet

import pytorch_lightning as pl
import itertools
import torchvision
from functools import partial
import numpy as np
from collections import OrderedDict
from einops import repeat
from typing import Dict, List, Union

from earthnet.util import get_obj_from_str

from .multimae_utils import trunc_normal_, Block


class EarthNetv1(pl.LightningModule):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        adapters_config,
        img_size=224,
        patch_size=16,
        num_global_tokens=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        # num_frames=16,
        # t_patch_size=4,
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        log_step=50,
        learning_rate=1e-4,
        drop_rate=0.0,
        mask_inputs=True,
        finetune=False,
        attn_drop_rate=0.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        num_encoded_tokens=64,
        ckpt_path=None,
        **kwargs,
    ):
        super().__init__()

        # Initialize input and output adapters
        # Adapters should handle normalization and denormalization
        self.adapters = dict()
        domains = np.array(list(adapters_config.keys()))
        for domain, cfg in adapters_config.items():
            self.adapters[domain] = partial(get_obj_from_str(cfg.target), **cfg.params)
            other_domains = domains[domains != domain]
            self.adapters[domain] = self.adapters[domain](
                patch_size_full=patch_size,
                dim_tokens=embed_dim,
                image_size=img_size,
                # t_patch_size=t_patch_size,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                # num_frames=num_frames,
                task=domain,
                context_tasks=other_domains,
            )

        self.adapters = nn.ModuleDict(self.adapters)

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.mask_inputs = mask_inputs
        self.finetune = finetune
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, embed_dim))
        trunc_normal_(self.global_tokens, std=0.02)

        self.log_step = log_step

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        # self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        self.learning_rate = learning_rate
        self.num_encoded_tokens = num_encoded_tokens
        # num_patches = self.patch_embed.num_patches
        # input_size = self.patch_embed.input_size
        # self.input_size = input_size

        # Transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.encoder = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.apply(self._init_weights)
        self.initialize_weights()

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        print("model initialized")

    def initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = np.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = np.sqrt(
                        6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def configure_optimizers(self):
        # params_list = list(self.parameters())
        params_list = [
            {"params": p, "lr": self.learning_rate}
            for name, p in self.named_parameters()
            if "transform." not in name  # exclude parameters with specific pattern
        ]
        optimizer = torch.optim.AdamW(
            params_list, lr=self.learning_rate, weight_decay=1e-6, betas=(0.5, 0.9)
        )
        return optimizer

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def nan_masking(self, x, namask):
        """
        Perform per-sample masking by missing data.
        Per-sample shuffling is done by argsort missing data count.
        len_keep is determined by the example in the batch with the most missing data.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        noise = torch.squeeze(namask)

        len_drop = torch.sum(noise > 0, dim=1).max().item()
        len_keep = int(L - len_drop)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def prepare_data_transformation(self, dataloader, n_batches=1000):
        for _, adapter in self.adapters.items():
            adapter.prepare_data_transformation(dataloader, n_batches=n_batches)
            print(_, adapter.transform.mu[_].data, adapter.transform.sd[_].data)

        for name, param in self.named_parameters():
            if ".transform" in name:
                param.requires_grad = False

    def sample_alphas(
        self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5
    ):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor(
            [list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:]
        )
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(
            valid_task_choices, 0, rand_per_sample_choice
        )
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
    ):
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        input_tokens_values = list(input_tokens.values())
        B = input_tokens_values[0].shape[0]
        device = input_tokens_values[0].device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        # alphas = [10., 1]
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()

        # handle rounding
        diff = num_encoded_tokens - samples_per_task.sum(dim=1)
        samples_per_task[:, 0] = samples_per_task[:, 0] + diff

        task_masks = []
        ids_aranges = []
        num_tokens_per_task = [
            task_tokens.shape[1] for task_tokens in input_tokens.values()
        ]

        tokens_finite = [
            torch.isfinite(task_tokens).all(axis=2)
            for task_tokens in input_tokens.values()
        ]
        sample_num_tokens_finite = torch.cat(
            [torch.sum(tk, axis=1).unsqueeze(1) for tk in tokens_finite], dim=1
        )

        # tokens_finite = torch.isfinite(input_tokens_arr).all(axis=2)
        for i in range(samples_per_task.shape[1]):
            diff = sample_num_tokens_finite - samples_per_task
            diff[diff > 0] = 0
            if torch.all(diff == 0):
                break
            samples_per_task = samples_per_task + (diff - torch.roll(diff, 1, 1))

        # tasks = list(input_tokens.keys())
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            noise = torch.where(tokens_finite[i], noise, 9999.0)

            ids_arange_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove

            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            # mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)

            ids_restore = torch.argsort(ids_arange_shuffle, dim=1)
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # print(i, samples_per_task[:,i], (torch.isfinite(input_tokens_values[i]).all(dim=2) * (mask == 0)).sum(dim=1))
            task_masks.append(mask)
            ids_aranges.append(ids_arange_shuffle)

        # tokens_all = torch.cat(input_tokens_values, dim=1)
        mask_all = torch.cat(task_masks, dim=1)

        # breakpoint()

        # ids_aranges_all = torch.cat(ids_aranges, dim=1)
        # ids_shuffle = ids_aranges_all  # torch.argsort(
        # ids_shuffle = torch.argsort(mask_all + 0.0 * torch.rand_like(mask_all.float()), dim=1,
        #        stable=True)
        ids_shuffle = torch.argsort(mask_all, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        # mask_all_2 = torch.ones_like(mask_all)
        # mask_all_2[:, :num_encoded_tokens] = 0

        # Unshuffle to get the binary mask
        # mask_all_2 = mask_all #torch.gather(mask_all_2, dim=1, index=ids_restore)

        # Split to get task masks
        # task_masks = torch.split(mask_all_2, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {
            domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)
        }

        input_tokens = torch.cat(
            [task_tokens for task_tokens in input_tokens.values()], dim=1
        )

        # Apply mask
        input_tokens = torch.gather(
            input_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]),
        )

        return task_masks, ids_keep, ids_restore

    def generate_finetune_masks(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
    ):
        """
        Sample a total of num_encoded_tokens from all tasks.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device
        num_tokens_per_task = [
            task_tokens.shape[1] for task_tokens in input_tokens.values()
        ]

        input_tokens_arr = torch.cat(
            [task_tokens for task_tokens in input_tokens.values()], dim=1
        )
        num_tokens = input_tokens_arr.shape[1]
        tokens_finite = torch.isfinite(input_tokens_arr).all(axis=2)
        # num_finite_tokens = tokens_finite.sum(axis=1, keepdim=True)

        noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
        noise = torch.where(tokens_finite, noise, 1.0)
        ids_arange_shuffle = torch.argsort(noise, dim=1)
        mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
        mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
        mask = torch.where(mask < num_encoded_tokens, 0, 1)

        mask_all = mask

        ids_shuffle = ids_arange_shuffle  # torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0

        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {
            domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)
        }

        return task_masks, ids_keep, ids_restore

    @staticmethod
    def make_mask(
        N_H,
        N_W,
        xy_idxs,
        full_tasks=[],
        indicate_visible=True,
        flatten=True,
        device="cuda",
    ):
        """
        Creates masks for each task, given lists of un-masked x,y coordinates.
        """
        xy_idxs = {k: torch.LongTensor(v) for k, v in xy_idxs.items()}

        task_masks = {k: torch.ones(N_H, N_W).to(device) for k in xy_idxs.keys()}

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        j = 0
        input_info["tasks"] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens_finite = torch.isfinite(tensor).all(axis=2).sum(axis=1)
            num_tokens = tensor.shape[1]
            d = {
                "num_tokens": num_tokens,
                "has_3d_posemb": False,  # TODO: Modify when adding non-2D tasks
                "has_2d_posemb": False,
                "start_idx": i,
                "end_idx": i + num_tokens,
                "num_tokens_finite": num_tokens_finite,
                # "size": task_sizes[domain]
            }
            if "PatchedAdapter3D" == self.adapters[domain].__class__.__name__:
                d["has_3d_posemb"] = True
            elif "PatchedAdapter2D" == self.adapters[domain].__class__.__name__:
                d["has_2d_posemb"] = True

            i += num_tokens
            j += num_tokens_finite
            input_info["tasks"][domain] = d

        input_info["image_size"] = image_size
        input_info["num_task_tokens"] = i
        input_info["num_task_tokens_finite"] = j
        input_info["num_global_tokens"] = self.num_global_tokens

        return input_info

    def data_transform(self, x):
        # Apply transformations to input data
        for domain in x:
            if self.adapters[domain].transform:
                s = {domain: x[domain]}
                x[domain] = self.adapters[domain].transform(s)[domain]
        return x

    def backward_transform(self, x):
        for domain in x:
            if self.adapters[domain].transform:
                s = {domain: x[domain]}
                x[domain] = self.adapters[domain].transform.backward(s)[domain]
        return x

    def forward(
        self,
        x,
        # mask_inputs=True,
        task_masks=None,
        num_encoded_tokens=128,
        alphas=1.0,
        sample_tasks_uniformly=False,
        fp32_output_adapters=[],
    ):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        :param alphas: Dirichlet distribution parameter alpha for task sampling.
            Higher alpha = harder, less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
            before Dirichlet sampling decides share of masked tokens between them.
        :param fp32_output_adapters: List of task identifiers to force output adapters to
            run with mixed precision turned off for stability reasons.
        """

        # Processing input modalities

        # Need image size for tokens->image reconstruction
        if "goes16" in x:
            B, C, T, H, W = x["goes16"].shape
        else:
            B, C, T, H, W = list(x.values())[
                0
            ].shape  # TODO: Deal with case where not all have same shape

        # task_sizes = {t: v.shape for t, v in x.items()}

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.adapters[domain].encode(tensor)
            for domain, tensor in x.items()
            if domain in self.adapters
        }

        input_info = self.generate_input_info(
            input_task_tokens=input_task_tokens,
            image_size=(T, H, W)
            # task_sizes=task_sizes,
        )

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if self.mask_inputs:
            num_encoded_tokens = (
                num_encoded_tokens
                if num_encoded_tokens is not None
                else self.num_encoded_tokens
            )
        else:
            num_encoded_tokens = sum(
                [tensor.shape[1] for tensor in input_task_tokens.values()]
            )

        task_finite_tokens = []
        for tensor in input_task_tokens.values():
            # n = torch.min(torch.sum(torch.isfinite(tensor).all(dim=2), dim=1))
            # num_encoded_tokens += n
            task_finite_tokens.append(
                torch.sum(torch.isfinite(tensor).all(dim=2), dim=1)
            )

        max_num_encoded_tokens = torch.min(sum(task_finite_tokens))
        num_encoded_tokens = min([num_encoded_tokens, max_num_encoded_tokens])

        # Generating masks
        if self.finetune and (task_masks is None):
            task_masks, ids_keep, ids_restore = self.generate_finetune_masks(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
            )
        elif task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                # task_masks, ids_keep, ids_restore = self.generate_finetune_masks(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
            )
        else:
            mask_all = torch.cat(
                [task_masks[task] for task in input_task_tokens.keys()], dim=1
            )
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, : (mask_all == 0).sum()]

        input_tokens = torch.cat(
            [task_tokens for task_tokens in input_task_tokens.values()], dim=1
        )

        # Apply mask
        input_tokens = torch.gather(
            input_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]),
        )

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, "() n d -> b n d", b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        encoder_tokens = self.encoder(input_tokens)
        input_nans = torch.isfinite(input_tokens).float().mean(axis=(1, 2))

        if (input_nans < (1 - 1e-6)).any():
            print("nans in input", input_nans)
            breakpoint()

        # Output decoders
        # if self.output_adapters is None:
        #     return encoder_tokens, task_masks

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.adapters[domain].decoder(
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.adapters
            if domain not in fp32_output_adapters
        }

        preds_nans = torch.isfinite(preds["goes16"]).float().mean(axis=(1, 2, 3, 4))
        if (preds_nans != 1).any():
            print("goes prediction has nans")
            breakpoint()

        # Force running selected output adapters in fp32 mode
        with torch.cuda.amp.autocast(enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

        return preds, task_masks

    def training_step(self, batch, batch_idx):

        batch = self.data_transform(batch)
        # batch = {key: torch.nan_to_num(val) for key, val in batch.items()}

        preds, tasks_masks = self.forward(
            batch, num_encoded_tokens=self.num_encoded_tokens
        )

        losses = {}
        for task, tensor in preds.items():
            if self.finetune:
                tasks_masks[task] = torch.ones_like(tasks_masks[task])
            losses[task] = self.adapters[task].loss(
                tensor, batch[task], mask=tasks_masks[task]
            )
            self.log(
                f"train/loss_{task}",
                losses[task],
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        loss = sum(losses.values())
        # print(losses)
        # print(torch.isnan(preds['goes16']).any(), torch.isnan(preds['goes16']).all())
        # breakpoint()

        if loss == 0:
            print("Loss is zero")
            import sys

            sys.exit()

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if self.global_step % self.log_step == 0:
            n_images = 8
            for i, (task, tensor) in enumerate(batch.items()):
                if len(tensor.shape) == 4:
                    self.log_images(
                        f"train/samples/inputs_{task}", tensor[:n_images, 0:1, :, :]
                    )
                    self.log_images(
                        f"train/samples/recons_{task}", preds[task][:n_images, 0:, :, :]
                    )
                elif len(tensor.shape) == 5:
                    self.log_images(
                        f"train/samples/inputs_{task}", tensor[:n_images, 0:1, 0, :, :]
                    )
                    self.log_images(
                        f"train/samples/recons_{task}",
                        preds[task][:n_images, 0:1, 0, :, :],
                    )

                    self.log_images(
                        f"train/sequence/inputs_{task}",
                        tensor[0, 0:1, :, :, :].transpose(0, 1),
                    )
                    self.log_images(
                        f"train/sequence/recons_{task}",
                        preds[task][0, 0:1, :, :, :].transpose(0, 1),
                    )

        return loss

    def validation_step(self, batch, batch_idx):

        batch = self.data_transform(batch)

        preds, tasks_masks = self.forward(
            batch, num_encoded_tokens=self.num_encoded_tokens
        )

        losses = {}
        for task, tensor in preds.items():
            if self.finetune:
                tasks_masks[task] = torch.ones_like(tasks_masks[task])
            losses[task] = self.adapters[task].loss(
                tensor, batch[task], mask=tasks_masks[task]
            )
            self.log(
                f"valid/loss_{task}",
                losses[task],
                logger=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        loss = sum(losses.values())

        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.global_step % self.log_step == 0:
            n_images = 8
            for i, (task, tensor) in enumerate(batch.items()):
                if len(tensor.shape) != 5:
                    continue
                self.log_images(
                    f"valid/samples/inputs_{task}", tensor[:n_images, 0:1, 0, :, :]
                )
                self.log_images(
                    f"valid/samples/recons_{task}", preds[task][:n_images, 0:1, 0, :, :]
                )

                self.log_images(
                    f"valid/sequence/inputs_{task}",
                    tensor[0, 0:1, :, :, :].transpose(0, 1),
                )
                self.log_images(
                    f"valid/sequence/recons_{task}",
                    preds[task][0, 0:1, :, :, :].transpose(0, 1),
                )

        return loss

    def log_images(self, name, sample_images):
        try:
            if self.global_step % self.log_step == 0:
                grid = torchvision.utils.make_grid(sample_images, nrow=4)
                xmn = nanmin(grid)
                xmx = nanmax(grid)
                grid = (grid - xmn) / (xmx - xmn)
                self.logger.experiment.add_image(name, grid, self.global_step)
        except AttributeError as err:
            print(f"Logger error. {err}")
            return

    # def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
    #    norms = grad_norm(self.encoder, norm_type=2)
    #    self.log_dict(norms)
    #    norms = grad_norm(self.adapters, norm_type=2)
    #    self.log_dict(norms)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


def nanmax(tensor):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max()
    return output


def nanmin(tensor):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min()
    return output
