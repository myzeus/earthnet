# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .multimae_utils import (
    build_2d_sincos_posemb,
    pair,
    trunc_normal_,
    build_3d_sincos_posemb,
    Block,
)


class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int, int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
    ):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (
            self.image_size[1] // patch_size_full
        )

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            if self.learnable_pos_emb:
                self.pos_emb = nn.Parameter(
                    self.pos_emb, requires_grad=self.learnable_pos_emb
                )
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.dim_tokens, h_posemb, w_posemb)
            )
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W),
            stride=(self.P_H, self.P_W),
        )

        self.encoder_transformer = nn.Sequential(
            *[
                Block(
                    dim=self.dim_tokens,
                    num_heads=6,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                )
            ]
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb"}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"
        assert (H % self.P_H == 0) and (
            W % self.P_W == 0
        ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}"
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), "b d nh nw -> b (nh nw) d")

        # Create positional embedding
        x_pos_emb = F.interpolate(
            self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False
        ).to(x.device)
        x_pos_emb = rearrange(x_pos_emb, "b d nh nw -> b (nh nw) d")

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        # Apply transformer encoder, masking nans
        mask = torch.isfinite(x)
        x[~mask] = 0.0
        x = self.encoder_transformer(x)
        x[~mask] = np.nan

        return x


class PatchedInputAdapter3D(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int, int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
        num_frames: int = 8,
        t_patch_size: int = 1,
        encoder_transformer: bool = False,
    ):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (
            (self.image_size[0] // patch_size_full)
            * (self.image_size[1] // patch_size_full)
            * (num_frames // t_patch_size)
        )
        self.num_frames = num_frames
        self.encoder_transformer = encoder_transformer

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)
        self.P_T = t_patch_size

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        t_posemb = self.num_frames // self.P_T

        if self.sincos_pos_emb:
            self.pos_emb = build_3d_sincos_posemb(
                h=h_posemb, w=w_posemb, t=t_posemb, embed_dim=self.dim_tokens
            )
            if self.learnable_pos_emb:
                self.pos_emb = nn.Parameter(
                    self.pos_emb, requires_grad=self.learnable_pos_emb
                )
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.dim_tokens, t_posemb, h_posemb, w_posemb)
            )
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=self.dim_tokens,
            kernel_size=(self.P_T, self.P_H, self.P_W),
            stride=(self.P_T, self.P_H, self.P_W),
        )

        if self.encoder_transformer:
            self.encoder = nn.Sequential(
                *[
                    Block(
                        dim=self.dim_tokens,
                        num_heads=6,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        drop=0.0,
                    )
                ]
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb"}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, T, H, W = x.shape
        finite_mask = torch.isfinite(x)
        x = torch.nan_to_num(x)

        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"
        assert (
            (H % self.P_H == 0) and (W % self.P_W == 0) and (T % self.P_T == 0)
        ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}"
        N_T, N_H, N_W = (
            T // self.P_T,
            H // self.P_H,
            W // self.P_W,
        )  # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), "b d nt nh nw -> b (nt nh nw) d")

        # Create positional embedding
        x_pos_emb = F.interpolate(
            self.pos_emb, size=(N_T, N_H, N_W), mode="trilinear", align_corners=False
        ).to(x.device)
        x_pos_emb = rearrange(x_pos_emb, "b d nt nh nw -> b (nt nh nw) d")

        finite_mask = F.interpolate(
            finite_mask.float(),
            size=(N_T, N_H, N_W),
            mode="trilinear",
            align_corners=False,
        )
        finite_mask = rearrange(finite_mask, "b d nt nh nw -> b (nt nh nw) d")
        finite_mask = finite_mask == 1  # .float()
        finite_mask = finite_mask.all(dim=-1).unsqueeze(-1)
        finite_mask = torch.repeat_interleave(finite_mask, x_patch.shape[-1], dim=-1)

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb
        # x = x * finite_mask

        if self.encoder_transformer:
            mask = torch.isfinite(x)
            x[~mask] = 0.0
            x = self.encoder(x)
            x[~mask] = np.nan

        x = torch.where(finite_mask == 1, x, torch.ones_like(x) * float("nan"))
        # print(x[0])
        # breakpoint()

        return x  # , finite_mask
