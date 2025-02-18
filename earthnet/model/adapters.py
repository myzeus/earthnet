import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from collections import OrderedDict

from .input_adapters import (
    PatchedInputAdapter,
    PatchedInputAdapter3D,
)
from .output_adapters import SpatialOutputAdapter3D, SpatialOutputAdapter

from earthnet.util import instantiate_from_config
from earthnet.losses import MaskedMSELoss, MaskedL1Loss
from earthnet.distributions import DiagonalGaussianDistribution


class PatchedAdapter3D(pl.LightningModule):
    def __init__(
        self,
        num_channels,
        num_frames,
        patch_size_full=16,
        image_size=224,
        dim_tokens=256,
        t_patch_size=1,
        stride_level=1,
        decoder_dim=240,
        decoder_depth=12,
        decoder_num_heads=12,
        decoder_use_task_queries=True,
        decoder_use_xattn=True,
        data_transform_config=None,
        task=None,
        context_tasks=None,
        learning_rate=1e-4,
        kl_weight=1e-4,
        log_step=100,
        latent_dim=32,
        ckpt_path=None,
        encoder_transformer=False,
    ):
        super().__init__()

        self.transform = None
        self.task = task
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.log_step = log_step

        if data_transform_config is not None:
            self.transform = instantiate_from_config(data_transform_config)

        self.encoder = PatchedInputAdapter3D(
            num_channels=num_channels,
            patch_size_full=patch_size_full,
            dim_tokens=dim_tokens,
            image_size=image_size,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
            stride_level=stride_level,
            sincos_pos_emb=True,
            learnable_pos_emb=False,
            encoder_transformer=encoder_transformer,
        )

        self.latent_dim = latent_dim
        self.q_0 = nn.Linear(dim_tokens, latent_dim * 2)
        self.post_q_0 = nn.Linear(latent_dim, dim_tokens)

        self.decoder = SpatialOutputAdapter3D(
            num_channels=num_channels,
            t_patch_size=t_patch_size,
            num_frames=num_frames,
            patch_size_full=patch_size_full,
            stride_level=stride_level,
            dim_tokens=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            use_task_queries=decoder_use_task_queries,
            task=task,
            dim_tokens_enc=dim_tokens,
            context_tasks=list(context_tasks),
            use_xattn=decoder_use_xattn,
        )

        self.loss = MaskedMSELoss(
            patch_size=patch_size_full, stride=stride_level, t_patch_size=t_patch_size
        )

        if ckpt_path is not None:
            ignore_keys = ["decoder.pos_emb"]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def prepare_data_transformation(self, dataloader, n_batches=1000):
        if self.transform:
            print("Training transform", self.task)
            samples = dict()
            for i, sample in enumerate(dataloader):
                if i > n_batches:
                    break
                if i == 0:
                    samples = {self.task: [sample[self.task]]}
                else:
                    samples[self.task].append(sample[self.task])

            samples[self.task] = torch.cat(samples[self.task], 0)
            self.transform.training_step(samples, 0)
            print(
                self.task,
                self.transform.mu[self.task].data,
                self.transform.sd[self.task].data,
            )

    def data_transform(self, x):
        return self.transform(x)

    def encode(self, x):
        enc = self.encoder(x)
        return enc

    def decode(self, x):
        return self.decoder(x)

    def get_empty_input_info(self, tokens, image_size):
        input_info = OrderedDict()
        input_info["tasks"] = {}
        input_info["tasks"][self.task] = {
            "num_tokens": tokens.shape[1],
            "has_3d_posemb": True,
            "start_idx": 0,
            "end_idx": tokens.shape[1],
        }
        input_info["image_size"] = image_size
        input_info["num_task_tokens"] = tokens.shape[1]
        input_info["num_global_tokens"] = tokens.shape[1]
        return input_info

    def forward(self, x):
        image_size = x.shape[2:]
        enc_tokens = self.encode(x)

        device = x.device
        B, num_tokens, dim_tokens = enc_tokens.shape
        tokens_finite = torch.isfinite(enc_tokens).all(axis=2)
        num_tokens_finite = tokens_finite.sum(axis=1)

        num_tokens_dec = min(num_tokens_finite)

        noise = torch.rand(B, num_tokens, device=device)
        noise = torch.where(tokens_finite, noise, 1.0)
        ids_arange_shuffle = torch.argsort(noise, dim=1)

        mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
        mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
        mask = torch.where(mask < num_tokens_dec, 0, 1)
        # mask = torch.where(mask < num_tokens_dec, 0, 1)

        # if mask.sum() == 0:
        #    print("mask is equal to zero")
        #    breakpoint()

        ids_restore = torch.argsort(ids_arange_shuffle, dim=1)
        ids_keep = ids_arange_shuffle[:, :num_tokens_dec]

        input_info = self.get_empty_input_info(enc_tokens, image_size)
        # ids_keep = torch.arange(0, enc_tokens.shape[1]).to(x.device)
        # ids_keep = ids_keep.repeat((x.shape[0], 1))
        # ids_restore = torch.arange(0, enc_tokens.shape[1]).to(x.device)

        # apply mask
        enc_tokens = torch.gather(
            enc_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, enc_tokens.shape[2]),
        )

        mu_logvar = self.q_0(enc_tokens)
        mu = mu_logvar[:, :, : self.latent_dim]
        logvar = mu_logvar[:, :, self.latent_dim :]
        posterior = DiagonalGaussianDistribution(mu, logvar)
        enc_tokens = self.post_q_0(posterior.sample())

        dec = self.decoder(
            encoder_tokens=enc_tokens,
            input_info=input_info,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
        )
        return dec, mask, enc_tokens, posterior

    def kl_loss(self, mu, logvar):
        mask = torch.isfinite(mu)
        num_finite = mask.sum(axis=[1, 2])
        mu = torch.nan_to_num(mu)
        logvar = torch.nan_to_num(logvar)
        s = torch.sum((-(mu**2) - logvar.exp() + 1 + logvar) * mask, dim=[1, 2])
        s /= num_finite
        s *= 0.5
        return -s.mean()

    def training_step(self, batch, batch_idx):
        batch = self.data_transform(batch)
        y, mask, enc_tokens, posterior = self(batch[self.task])
        mask = torch.ones_like(mask)
        recon_loss = self.loss(y, batch[self.task], mask=mask)
        # kl_loss = self.kl_loss(enc_tokens, enc_logvar) * self.kl_weight
        kl_loss = self.kl_loss(posterior.mean, posterior.logvar) * self.kl_weight

        if recon_loss == 0:
            print("Loss equals 0, quitting")
            import sys

            sys.exit()

        loss = recon_loss + kl_loss

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/kl",
            kl_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        n_images = 4
        if self.global_step % self.log_step == 0:
            self.log_images(
                f"train/samples/inputs_{self.task}",
                batch[self.task][:n_images, 0:1, 0, :, :],
            )
            self.log_images(
                f"train/samples/recons_{self.task}", y[:n_images, 0:1, 0, :, :]
            )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.data_transform(batch)
        y, mask, enc_tokens, posterior = self(batch[self.task])
        mask = torch.ones_like(mask)
        loss = self.loss(y, batch[self.task], mask=mask)
        # kl_loss = self.kl_loss(enc_tokens, enc_logvar) * self.kl_weight
        kl_loss = self.kl_loss(posterior.mean, posterior.logvar) * self.kl_weight
        loss += kl_loss
        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "valid/kl",
            kl_loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

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


class PatchedAdapter2D(pl.LightningModule):
    def __init__(
        self,
        num_channels,
        # num_frames,
        patch_size_full=16,
        image_size=224,
        dim_tokens=256,
        # t_patch_size=1,
        stride_level=1,
        decoder_dim=240,
        decoder_depth=12,
        decoder_num_heads=12,
        decoder_use_task_queries=True,
        decoder_use_xattn=True,
        data_transform_config=None,
        task=None,
        context_tasks=None,
        learning_rate=1e-4,
        kl_weight=1e-4,
        log_step=100,
        latent_dim=32,
        ckpt_path=None,
    ):
        super().__init__()

        self.transform = None
        self.task = task
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.log_step = log_step

        if data_transform_config is not None:
            self.transform = instantiate_from_config(data_transform_config)

        self.encoder = PatchedInputAdapter(
            num_channels=num_channels,
            patch_size_full=patch_size_full,
            dim_tokens=dim_tokens,
            image_size=image_size,
            # num_frames=num_frames,
            # t_patch_size=t_patch_size,
            stride_level=stride_level,
            sincos_pos_emb=True,
            learnable_pos_emb=False,
        )

        self.latent_dim = latent_dim
        self.q_0 = nn.Linear(dim_tokens, latent_dim * 2)
        self.post_q_0 = nn.Linear(latent_dim, dim_tokens)

        self.decoder = SpatialOutputAdapter(
            num_channels=num_channels,
            # t_patch_size=t_patch_size,
            # num_frames=num_frames,
            patch_size_full=patch_size_full,
            stride_level=stride_level,
            dim_tokens=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            use_task_queries=decoder_use_task_queries,
            task=task,
            dim_tokens_enc=dim_tokens,
            context_tasks=list(context_tasks),
            use_xattn=decoder_use_xattn,
        )

        self.loss = MaskedL1Loss(
            patch_size=patch_size_full,
            stride=stride_level,  # , t_patch_size=t_patch_size
        )

        if ckpt_path is not None:
            ignore_keys = []
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def prepare_data_transformation(self, dataloader, n_batches=1000):
        if self.transform:
            print("Training transform", self.task)
            samples = dict()
            for i, sample in enumerate(dataloader):
                if i > n_batches:
                    break
                if i == 0:
                    samples = {self.task: [sample[self.task]]}
                else:
                    samples[self.task].append(sample[self.task])

            samples[self.task] = torch.cat(samples[self.task], 0)
            self.transform.training_step(samples, 0)
            print(
                self.task,
                self.transform.mu[self.task].data,
                self.transform.sd[self.task].data,
            )

    def data_transform(self, x):
        return self.transform(x)

    def encode(self, x):
        enc = self.encoder(x)
        return enc

    def decode(self, x):
        return self.decoder(x)

    def get_empty_input_info(self, tokens, image_size):
        input_info = OrderedDict()
        input_info["tasks"] = {}
        input_info["tasks"][self.task] = {
            "num_tokens": tokens.shape[1],
            "has_2d_posemb": True,
            "start_idx": 0,
            "end_idx": tokens.shape[1],
        }
        input_info["image_size"] = image_size
        input_info["num_task_tokens"] = tokens.shape[1]
        input_info["num_global_tokens"] = tokens.shape[1]
        return input_info

    def forward(self, x):
        image_size = x.shape[2:]
        enc_tokens = self.encode(x)

        device = x.device
        B, num_tokens, dim_tokens = enc_tokens.shape
        tokens_finite = torch.isfinite(enc_tokens).all(axis=2)
        num_tokens_finite = tokens_finite.sum(axis=1)

        num_tokens_dec = min(num_tokens_finite)

        noise = torch.rand(B, num_tokens, device=device)
        noise = torch.where(tokens_finite, noise, 1.0)
        ids_arange_shuffle = torch.argsort(noise, dim=1)

        mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
        mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
        mask = torch.where(mask < num_tokens_dec, 0, 1)
        # mask = torch.where(mask < num_tokens_dec, 0, 1)

        # if mask.sum() == 0:
        #    print("mask is equal to zero")
        #    breakpoint()

        ids_restore = torch.argsort(ids_arange_shuffle, dim=1)
        ids_keep = ids_arange_shuffle[:, :num_tokens_dec]

        input_info = self.get_empty_input_info(enc_tokens, image_size)
        # ids_keep = torch.arange(0, enc_tokens.shape[1]).to(x.device)
        # ids_keep = ids_keep.repeat((x.shape[0], 1))
        # ids_restore = torch.arange(0, enc_tokens.shape[1]).to(x.device)

        # apply mask
        enc_tokens = torch.gather(
            enc_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, enc_tokens.shape[2]),
        )

        mu_logvar = self.q_0(enc_tokens)
        mu = mu_logvar[:, :, : self.latent_dim]
        logvar = mu_logvar[:, :, self.latent_dim :]
        posterior = DiagonalGaussianDistribution(mu, logvar)
        enc_tokens = self.post_q_0(posterior.sample())

        dec = self.decoder(
            encoder_tokens=enc_tokens,
            input_info=input_info,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
        )
        return dec, mask, enc_tokens, posterior

    def kl_loss(self, mu, logvar):
        mask = torch.isfinite(mu)
        num_finite = mask.sum(axis=[1, 2])
        mu = torch.nan_to_num(mu)
        logvar = torch.nan_to_num(logvar)
        s = torch.sum((-(mu**2) - logvar.exp() + 1 + logvar) * mask, dim=[1, 2])
        s /= num_finite
        s *= 0.5
        return -s.mean()

    def training_step(self, batch, batch_idx):
        batch = self.data_transform(batch)
        y, mask, enc_tokens, posterior = self(batch[self.task])
        mask = torch.ones_like(mask)
        loss = self.loss(y, batch[self.task], mask=mask)
        # kl_loss = self.kl_loss(enc_tokens, enc_logvar) * self.kl_weight
        kl_loss = self.kl_loss(posterior.mean, posterior.logvar) * self.kl_weight

        if loss == 0:
            print("Loss equals 0, quitting")
            breakpoint()
            import sys

            sys.exit()

        loss += kl_loss
        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "train/kl",
            kl_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        n_images = 4
        if self.global_step % self.log_step == 0:
            self.log_images(
                f"train/samples/inputs_{self.task}",
                batch[self.task][:n_images, 0:1, :, :],
            )
            self.log_images(
                f"train/samples/recons_{self.task}", y[:n_images, 0:1, :, :]
            )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.data_transform(batch)
        y, mask, enc_tokens, posterior = self(batch[self.task])
        mask = torch.ones_like(mask)
        loss = self.loss(y, batch[self.task], mask=mask)
        # kl_loss = self.kl_loss(enc_tokens, enc_logvar) * self.kl_weight
        kl_loss = self.kl_loss(posterior.mean, posterior.logvar) * self.kl_weight
        loss += kl_loss
        self.log(
            "valid/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "valid/kl",
            kl_loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

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
