from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import (
    ConvNeXt,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
)


def convnext_factory(
    model_name: Literal[
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
    ],
    weights: bool = False,
) -> Tuple[ConvNeXt, Dict[str, Any]]:
    if model_name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights if weights else None)
        config = {"feature_dim": 768}
    elif model_name == "convnext_small":
        model = convnext_small(weights=ConvNeXt_Small_Weights if weights else None)
        config = {"feature_dim": 768}
    elif model_name == "convnext_base":
        model = convnext_base(weights=ConvNeXt_Base_Weights if weights else None)
        config = {"feature_dim": 1024}
    elif model_name == "convnext_large":
        model = convnext_large(weights=ConvNeXt_Large_Weights if weights else None)
        config = {"feature_dim": 1536}
    return model, config
