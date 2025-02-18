# --------------------------------------------------------
# Based on MultiMAE, timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/EPFL-VILAB/MultiMAE 
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedMSELoss(nn.Module):
    """L2 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, t_patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.t_patch_size = t_patch_size

    def forward(self, input, target, mask=None):

        T, H, W = input.shape[-3:]
        nh, nw = H // self.scale_factor, W // self.scale_factor
        nt = T // self.t_patch_size

        finite_mask = torch.isfinite(target).all(dim=1)
        # input = torch.nan_to_num(input)
        target = torch.nan_to_num(target)

        loss = F.mse_loss(input, target, reduction="none")

        if mask is not None:
            # Resize mask and upsample
            mask = rearrange(mask, "b (nt nh nw) -> b nt nh nw", nh=nh, nw=nw, nt=nt)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(T, H, W), mode="nearest"
            ).squeeze(1)
            loss = loss.nanmean(dim=1)  # B, C, T, H, W -> B, T, H, W

            mask = mask * finite_mask

            # mask_flat = mask.flatten(start_dim=1).sum(dim=1)
            # if 0 in mask_flat:
            # print("Mask has a 0")
            # breakpoint()
            # return torch.tensor(0).to(loss.device)

            loss = loss * mask
            # print("loss in mask", torch.nansum(loss), "mask", torch.nanmean(mask), "values",
            #        torch.nanmean(target), "pred", torch.nanmean(input))

            # Compute mean per sample
            loss = loss.flatten(start_dim=1).nansum(dim=1) / (
                mask.flatten(start_dim=1).sum(dim=1) + 1e-2
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedL1Loss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(
            imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p
        )
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(
            x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p
        )
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        finite_mask = torch.isfinite(target).all(dim=1)
        # input = torch.nan_to_num(input)
        target = torch.nan_to_num(target)

        loss = F.l1_loss(input, target, reduction="none")

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W

            mask = mask * finite_mask
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).nansum(dim=1) / (
                mask.flatten(start_dim=1).sum(dim=1) + 1e-5
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
