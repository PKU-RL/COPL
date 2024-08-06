from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..others.mineclip_official.modules import QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.num_heads = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, mask: torch.Tensor = None):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if mask is None and self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            mask = self.attn_mask
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_qkv: bool = False):
        q, k, v = None, None, None
        if return_qkv:
            y = self.ln_1(x)
            y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
            L, N, C = y.shape
            y = y.view(L, N, 3, C//3).permute(2, 0, 1, 3).reshape(3*L, N, C//3)
            y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
            q, k, v = y.tensor_split(3, dim=0)  # LND
            v += x
            v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x, q, k, v


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self._layers = layers
        self.output_dim = output_dim
        self.num_heads = heads
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn(161, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bs, ts, c, h, w = x.shape
        x = x.reshape(bs*ts, c, h, w)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.blocks(x)
        
        # Modified transformer
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x, q, k, v = block(x, mask, return_qkv=True)
            else:
                x, _, _, _ = block(x, mask, return_qkv=False)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)

        x = self.ln_post(x[:, 0 ])
        v = self.ln_post(v[:, 1:])
        q = q[:, 1:]
        k = k[:, 1:]

        if self.projection is not None:
            x = x @ self.projection
            feat = v @ self.projection

        x = x.reshape(bs, ts, -1)
        
        h //= self._patch_size
        w //= self._patch_size
        feat = feat.reshape(bs, ts, h, w, -1)
        q = q.reshape(bs, ts, h, w, -1)
        k = k.reshape(bs, ts, h, w, -1)
        
        return x, feat, q, k

    def get_attn_mask(self, image_mask: np.ndarray) -> np.ndarray:
        b, h, w = image_mask.shape
        mask = ~image_mask.astype(bool)
        mask = np.concatenate([np.zeros((b, 1), dtype=bool), mask.reshape(b, -1)], axis=-1)
        mask = mask[:, None].repeat(1 + h*w, axis=1)        # B(1+HW) -> B(1+HW)(1+HW)
        mask = mask[:, None].repeat(self.num_heads, axis=1) # B(1+HW)(1+HW) -> BN(1+HW)(1+HW)
        mask = mask.reshape(b * self.num_heads, 1 + h*w, 1 + h*w)
        return mask
