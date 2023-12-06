from typing import Callable, Optional, Tuple, Union
from timm.layers import Mlp, PatchEmbed
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import (
    Attention,
    Block,
    Mlp,
    PatchEmbed,
    VisionTransformer,
    checkpoint_filter_fn,
    checkpoint_seq,
)
from functools import partial


from .stgm import STGM


class SPViT(VisionTransformer):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        window_size=4,
        # STGM location are the blocks that the STGM block overrides.
        stgm_locations=[5, 6],
        **kwargs
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            pre_norm,
            fc_norm,
            drop_rate,
            pos_drop_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            embed_layer,
            norm_layer,
            act_layer,
            block_fn,
            mlp_layer,
        )
        # Set this manually because 
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # This is the number of transformer blocks the STGM
        # module uses.
        self.stgm_locations = [i - 1 for i in stgm_locations]
        self.stgm = STGM(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop_rate,
            attn_drop_rate,
            init_values,
            drop_path_rate,
            act_layer,
            norm_layer,
            mlp_layer,
            window_size=window_size,
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i, block in enumerate(self.blocks):
                if i in self.stgm_locations:
                    if i == self.stgm_locations[0]:
                        x = self.stgm(x)
                    else:
                        continue
                else:
                    x = block(x)

            # x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


if __name__ == "__main__":
    spvit = SPViT()
    print(spvit)
    print(spvit(torch.randn(1,3, 224, 224)).shape)
