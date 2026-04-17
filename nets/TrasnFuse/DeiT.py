# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import math
from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


# __all__ = [
#     'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
#    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
#    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
#    'deit_base_distilled_patch16_384',
#]


class DeiT(VisionTransformer):
    """
    改动点：
    1) 支持 img_size 可变（例如 256），in_chans 可变（例如 1）
    2) forward 时对 pos_embed 做动态插值，避免 token 数不匹配
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 注意：这里仍然创建 pos_embed，但 forward 会按当前 token 网格动态插值
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))  # 直接用 patch token 长度

    def _resize_pos_embed(self, pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        pos_embed: (1, N0, C)
        返回: (1, h*w, C)
        """
        N0 = pos_embed.shape[1]
        C = pos_embed.shape[2]
        s = int(math.sqrt(N0))
        if s * s != N0:
            # 兜底：如果不是方形（一般不会发生），直接线性插值到目标长度
            return F.interpolate(pos_embed.transpose(1, 2), size=h*w, mode="linear", align_corners=False).transpose(1, 2)

        pe = pos_embed.transpose(1, 2).contiguous().view(1, C, s, s)      # (1,C,s,s)
        pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        pe = pe.flatten(2).transpose(1, 2).contiguous()                   # (1,h*w,C)
        return pe

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # 当前 token 网格大小
        N = x.shape[1]
        h = H // self.patch_embed.patch_size[0]
        w = W // self.patch_embed.patch_size[1]
        if h * w != N:
            # 极少数情况下（pad/crop）会不一致，兜底用 sqrt 推断
            s = int(math.sqrt(N))
            h, w = s, s

        pe = self._resize_pos_embed(self.pos_embed, h, w)  # (1,N,C)
        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


# @register_model
def deit_small_patch16_224(pretrained=False, img_size=256, in_chans=1, **kwargs):
    """
    改动点：
    - 不再固定 224；允许 img_size=256
    - 允许 in_chans=1
    - 不再写死 pos_embed 插值到 (12,16)
    """
    model = DeiT(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    # if pretrained:
    #    ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth', map_location="cpu")
    #    model.load_state_dict(ckpt['model'], strict=False)

    model.head = nn.Identity()
    return model


# @register_model
def deit_base_patch16_224(pretrained=False, img_size=256, in_chans=1, **kwargs):
    model = DeiT(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_224-b5f2ef4d.pth', map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)

    model.head = nn.Identity()
    return model


# @register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = DeiT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(ckpt["model"])

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    pe = F.interpolate(pe, size=(24, 32), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model