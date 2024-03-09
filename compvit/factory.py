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
from .models.mae import MAECompVit, Decoder, DINOHead

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
    teacher = torch.compile(teacher, mode="max-autotune", fullgraph=True)

    student, compvit_conf = compvit_factory(student_name)

    if decoder_conf['type'] == "transformer":
        decoder = Decoder(
            embed_dim=decoder_conf["decoder_dim"],
            depth=decoder_conf["num_layers"],
            num_heads=decoder_conf["nhead"],
            mlp_ratio=decoder_conf["mlp_ratio"],
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            act_layer=nn.GELU,
            block_fn=Block,
            ffn_layer="mlp",
            init_values=compvit_conf["init_values"],
        )
    elif conf['loss'] == "ce" and decoder_conf['type'] == "mlp":
        decoder = DINOHead(student.embed_dim, decoder_conf["decoder_dim"],)
    else:
        decoder = nn.Identity()

    return (
        MAECompVit(
            baseline=teacher,
            encoder=student,
            decoder=decoder,
            baseline_embed_dim=teacher.embed_dim,
            embed_dim=student.embed_dim,
            decoder_embed_dim=decoder_conf["decoder_dim"],
            loss=conf["loss"],
        ),
        {**dino_conf, **compvit_conf, **decoder_conf},
    )


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224).to("cuda")
    model, conf = compvit_factory("compvits14")
    model = model.to("cuda")
    print(model(x, is_training=True)['x_norm'].shape)

    x = torch.randn(1, 3, 224, 224).to("cuda")
    model, conf = mae_factory("dinov2_vits14", "compvits14")
    model = model.to("cuda")
    print(model(x, x))