from typing import Callable, Literal, Optional, Tuple, Union
from typing_extensions import Literal
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
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    List,
)

from timm.layers import LayerType
from torch.nn.modules import Module

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .stgm import STGM


class SPViT(VisionTransformer):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        window_size=4,
        # STGM location are the blocks that the STGM block overrides.
        stgm_locations=[5, 6],
        bottleneck=True,
        **kwargs
    ) -> None:
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
            reg_tokens,
            pre_norm,
            fc_norm,
            dynamic_img_size,
            dynamic_img_pad,
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

        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # This is used to control if we want to bottleneck.
        self.bottleneck = bottleneck

        # This is the number of transformer blocks the STGM
        # module uses.
        # Fix the index.
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
            num_patches=self.patch_embed.num_patches + self.num_prefix_tokens,
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        for i in self.stgm_locations:
            self.blocks[i] = None

        self.init_weights()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i, block in enumerate(self.blocks):
                if i in self.stgm_locations and self.bottleneck:
                    if i == self.stgm_locations[0]:
                        x = self.stgm(x)
                    else:
                        continue
                else:
                    x = block(x)

            # x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def peft_parameters(self, train_bottleneck=False, blocks=False):
        parameters = []

        if train_bottleneck:
            parameters.extend(self.stgm.parameters())

        if blocks:
            print("Training post-bottleneck Blocks")
            for i, block in enumerate(self.blocks):
                if i > self.stgm_locations[-1]:
                    parameters.extend(block.parameters())

        return parameters
