import os
from functools import partial
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf

from dinov2.layers import MemEffAttention
from dinov2.layers import NestedTensorBlock as Block

from .models.vision_transformer import DinoVisionTransformer

CONFIG_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "configs"


def dinov2_factory(
    model_name: Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
):
    config_path = CONFIG_PATH / "dinov2.yaml"
    conf = OmegaConf.load(config_path)
    return DinoVisionTransformer(
        block_fn=partial(Block, attn_class=MemEffAttention), **conf[model_name]
    )
