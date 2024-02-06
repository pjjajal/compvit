from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused
from dinov2.models.vision_transformer import DinoVisionTransformer

from ..layers.bottleneck import (
    mixer_bottleneck,
    mixer_bottleneck_relu,
    mixer_bottleneck_multi,
    mixer_bottleneck_multi_v2,
    conv_bottleneck
)
from ..layers.compressor import Compressor
from torch.profiler import profile, record_function, ProfilerActivity



class CompViT(DinoVisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        num_compressed_tokens=0,
        num_patches=256,
        bottleneck: Literal[
            "mixer_bottleneck",
            "mixer_bottleneck_relu",
            "mixer_bottleneck_multi",
            "mixer_bottleneck_multi_v2",
        ] = "mixer_bottleneck_v2",
        bottleneck_locs=[5, 6],
        bottleneck_size=1,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            ffn_bias,
            proj_bias,
            drop_path_rate,
            drop_path_uniform,
            init_values,
            embed_layer,
            act_layer,
            block_fn,
            ffn_layer,
            block_chunks,
            num_register_tokens,
            interpolate_antialias,
            interpolate_offset,
        )
        # Boilerplate from DINOv2 implementation
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.total_tokens = num_patches + self.num_tokens + self.num_register_tokens
        # self.total_tokens = num_patches
        self.num_compressed_tokens = num_compressed_tokens + 1 # Add CLS Token

        # Add compressor.
        if num_compressed_tokens:
            self.compress = True

            # Set the blocks where bottleneck will be with None
            self.bottleneck_locs = [i - 1 for i in bottleneck_locs]
            for i in self.bottleneck_locs:
                self.blocks[i] = None

            if bottleneck == "mixer_bottleneck":
                bottleneck = partial(mixer_bottleneck, dim=embed_dim)
            elif bottleneck == "mixer_bottleneck_relu":
                bottleneck = partial(mixer_bottleneck_relu, dim=embed_dim)
            elif bottleneck == "mixer_bottleneck_multi":
                bottleneck = partial(
                    mixer_bottleneck_multi, dim=embed_dim, ratio=mlp_ratio
                )
            elif bottleneck == "mixer_bottleneck_multi_v2":
                bottleneck = partial(
                    mixer_bottleneck_multi_v2,
                    dim=embed_dim,
                    ratio=mlp_ratio,
                    bottleneck_size=bottleneck_size,
                )
            elif bottleneck == "conv_bottleneck":
                bottleneck = partial(
                    conv_bottleneck,
                    dim=embed_dim,
                    ratio=mlp_ratio,
                    bottleneck_size=bottleneck_size,
                )

            self.compressor = Compressor(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                num_compressed_tokens=self.num_compressed_tokens,
                num_tokens=self.total_tokens,
                bottleneck=bottleneck,
            )

    def forward_features(self, x, masks=None, get_attn=False):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            if self.compress and i in self.bottleneck_locs:
                if i == self.bottleneck_locs[0]:
                    x = self.compressor(x, get_attn)
                else:
                    continue
            else:
                x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_norm": x_norm,
            "x_prenorm": x,
            "masks": masks,
        }


if __name__ == "__main__":
    print(
        CompViT(num_compressed_tokens=1, block_chunks=0, bottleneck="mixer_bottleneck")
    )
