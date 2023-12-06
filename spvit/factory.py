from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from timm.models._builder import build_model_with_cfg
from timm.models.deit import VisionTransformerDistilled
from timm.models.vision_transformer import checkpoint_filter_fn, checkpoint_seq

from .spvit_model import SPViT


def deit3_small_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_base_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_large_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    model_cls = VisionTransformerDistilled if distilled else SPViT
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs,
    )
    return model


class SPViTFactory(Enum):
    # DeiT-S
    deit3_small_patch16_224 = (
        deit3_small_patch16_224,
        {
            "pretrained": True,
            "class_token": False,
            "global_pool": "avg",
            "pretrained_strict": False,
            "init_values": 1e-4
        },
    )

    # DeiT-B
    deit3_base_patch16_224 = (
        deit3_base_patch16_224,
        {
            "pretrained": True,
            "class_token": False,
            "global_pool": "avg",
            "pretrained_strict": False,
            "init_values": 1e-4
        },
    )

    # DeiT-L
    deit3_large_patch16_224 = (
        deit3_large_patch16_224,
        {
            "pretrained": True,
            "class_token": False,
            "global_pool": "avg",
            "pretrained_strict": False,
            "init_values": 1e-4
        },
    )


if __name__ == "__main__":
    constructor, args = SPViTFactory["deit3_small_patch16_224"].value

    spvit = constructor(**args)
    print(spvit(torch.randn(1, 3, 224, 224)).shape)
