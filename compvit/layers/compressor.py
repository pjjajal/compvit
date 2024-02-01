from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import MemEffAttention, Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused

from .attention import CrossAttention
from .block import CompBlock


class Compressor(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_compressed_tokens: int = 16,
        num_tokens: int = 196,
        bottleneck: nn.Module = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.bottleneck = bottleneck(self.num_tokens, self.num_compressed_tokens)

        self.block_1 = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )
        self.block_2 = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )

        self.global_center = nn.Parameter(
            torch.zeros((1, self.num_compressed_tokens, dim)),
            requires_grad=True,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.global_center, std=1e-6)

    def forward(self, x, get_attn=False):
        B, N, C = x.shape

        # Compressing tokens
        compressed_tokens = self.bottleneck(x)

        # Transfer to compressed tokens
        x = torch.concat([x, compressed_tokens], dim=1)
        compressed_tokens = self.block_1(x, compressed_tokens, get_attn)

        # Refine compressed tokens
        # x = torch.concat([x, compressed_tokens], dim=1)
        compressed_tokens = self.block_2(x, compressed_tokens + self.global_center, get_attn)

        return compressed_tokens


if __name__ == '__main__':
    compressor = Compressor(384, 8, 4)
    print(compressor(torch.randn((1,196,384))).shape)