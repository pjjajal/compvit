import os
from functools import partial
from pathlib import Path
from typing import Literal

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
    model_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"]
):
    config_path = CONFIG_PATH / "compvit_dinov2.yaml"
    conf = OmegaConf.load(config_path)
    return CompViT(
        block_fn=partial(Block, attn_class=MemEffAttention), **conf[model_name]
    )


def mae_factory(
    teacher_name: Literal[
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    ],
    student_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"],
):
    config_path = CONFIG_PATH / "mae.yaml"
    conf = OmegaConf.load(config_path)

    decoder_conf = conf['decoder']

    teacher = dinov2_factory(teacher_name)

    student = compvit_factory(student_name)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=student.embed_dim,
        nhead=decoder_conf["nhead"],
        dim_feedforward=int(student.embed_dim * decoder_conf["mlp_ratio"]),
        dropout=0.0,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=True,
    )
    decoder = nn.TransformerDecoder(decoder_layer, decoder_conf["num_layers"])

    return MAECompVit(
        baseline=teacher,
        encoder=student,
        decoder=decoder,
        baseline_embed_dim=teacher.embed_dim,
        embed_dim=student.embed_dim,
        decoder_embed_dim=student.embed_dim,
        norm_layer=LayerNorm,
    )

if __name__ == "__main__":
    mae = mae_factory('dinov2_vits14', 'compvits14')
    print(mae)