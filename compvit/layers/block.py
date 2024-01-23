import torch
from torch import Tensor

from dinov2.layers.block import Block


class CompBlock(Block):
    def forward(self, x_or_x_list, compressed_tokens, get_attn=False):
        def attn_residual_func(x: Tensor, compressed_tokens: Tensor) -> Tensor:
            x = self.norm1(x)
            compressed_tokens = self.norm1(compressed_tokens)
            compressed_tokens, attn_map = self.attn(x, compressed_tokens, get_attn)
            return self.ls1(compressed_tokens)
            # return self.ls1(self.attn(x, compressed_tokens, get_attn))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.0:
            x = compressed_tokens + self.drop_path1(
                attn_residual_func(x_or_x_list, compressed_tokens)
            )
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = compressed_tokens + attn_residual_func(x_or_x_list, compressed_tokens)
            x = x + ffn_residual_func(x)
        return x
