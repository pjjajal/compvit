import os
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn import LayerNorm

from dinov2.factory import dinov2_factory
from dinov2.layers import MemEffAttention
from dinov2.layers import NestedTensorBlock as Block

from .models.compvit import CompViT
from .models.mae import MAECompVit

CONFIG_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "configs"


def compvit_factory(
    model_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"],
    **kwargs
):
    config_path = CONFIG_PATH / "compvit_dinov2.yaml"
    # Loads the default configuration.
    conf = OmegaConf.load(config_path)
    # kwargs can overwrite the default config. This allows for overriding config defaults.
    conf = OmegaConf.merge(conf[model_name], kwargs)

    return (
        CompViT(
            block_fn=partial(Block, attn_class=MemEffAttention), **conf
        ),
        conf,
    )


def mae_factory(
    teacher_name: Literal[
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    ],
    student_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"],
):
    config_path = CONFIG_PATH / "mae.yaml"
    conf = OmegaConf.load(config_path)

    decoder_conf = conf["decoder"]

    teacher, dino_conf = dinov2_factory(teacher_name)

    student, compvit_conf = compvit_factory(student_name)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=decoder_conf["decoder_dim"],
        nhead=decoder_conf["nhead"],
        dim_feedforward=int(student.embed_dim * decoder_conf["mlp_ratio"]),
        dropout=0.0,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=True,
    )
    decoder = nn.TransformerDecoder(decoder_layer, decoder_conf["num_layers"])

    return (
        MAECompVit(
            baseline=teacher,
            encoder=student,
            decoder=decoder,
            baseline_embed_dim=teacher.embed_dim,
            embed_dim=student.embed_dim,
            decoder_embed_dim=decoder_conf["decoder_dim"],
            norm_layer=LayerNorm,
            loss=conf["loss"],
        ),
        {**dino_conf, **compvit_conf, **decoder_conf},
    )


if __name__ == "__main__":
    compvit_factory("compvits14")
