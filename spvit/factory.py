from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from timm.models._builder import build_model_with_cfg
from timm.models.deit import VisionTransformerDistilled
from timm.models.vision_transformer import checkpoint_filter_fn, checkpoint_seq

from .spvit_model import SPViT


def vit_tiny_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """ViT-Tiny (Vit-Ti/16)"""
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )
    return model


def vit_small_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def vit_base_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def vit_large_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def _create_vision_transformer(
    variant: str, pretrained: bool = False, **kwargs
) -> SPViT:
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    if "flexi" in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(
            checkpoint_filter_fn, interpolation="bilinear", antialias=False
        )
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    # strict = True
    # if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
    #     strict = False
    strict = kwargs.pop("pretrained_strict")

    return build_model_with_cfg(
        SPViT,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


def deit_tiny_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        "deit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_small_patch16_224(pretrained=False, **kwargs) -> SPViT:
    """DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
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
    # ViT-Ti
    vit_tiny_patch16_224 = (
        vit_tiny_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )
    # ViT-S
    vit_small_patch16_224 = (
        vit_small_patch16_224,
        {
            "pretrained": True,
            # "class_token": True,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )
    # ViT-B
    vit_base_patch16_224 = (
        vit_base_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )
    # ViT-L
    vit_large_patch16_224 = (
        vit_large_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )

    # DeiT-Ti
    deit_tiny_patch16_224 = (
        deit_tiny_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )
    # DeiT-S
    deit3_small_patch16_224 = (
        deit3_small_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
            "init_values": 1e-4,
        },
    )

    # DeiT-B
    deit3_base_patch16_224 = (
        deit3_base_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )

    # DeiT-L
    deit3_large_patch16_224 = (
        deit3_large_patch16_224,
        {
            "pretrained": True,
            # "class_token": False,
            # "global_pool": "avg",
            "pretrained_strict": False,
        },
    )


def create_model(
    model_name, window_size=4, stgm_location=[5, 6], bottleneck=True, pretrained=True
):
    constructor, args = SPViTFactory[model_name].value
    args["window_size"] = window_size
    args["stgm_locations"] = stgm_location
    args["bottleneck"] = bottleneck
    args["pretrained"] = pretrained
    spvit = constructor(**args)

    return spvit


if __name__ == "__main__":
    # constructor, args = SPViTFactory["vit_tiny_patch16_224"].value

    # spvit = constructor(**args)
    spvit = create_model("deit_tiny_patch16_224", window_size=95)
    print(spvit)
    print(spvit.forward_features(torch.randn(1, 3, 224, 224)).shape)
