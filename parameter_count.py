from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from spvit.spvit_model import SPViT


TOY_EXPERIMENTS_PATH = Path("./toy_experiments")
CONFIG_PATH = TOY_EXPERIMENTS_PATH / "configs"

config_path = CONFIG_PATH / ("cifar100" + ".yaml")
configs = OmegaConf.load(config_path)

model_config = configs["vit"]
model = VisionTransformer(**model_config)
vit_total_p = sum(p.numel() for p in model.parameters())
print(f"vit:  {vit_total_p:,}")

model_config = configs["compvit"]
model = SPViT(**model_config)
compvit_total_p = sum(p.numel() for p in model.parameters())
bottleneck_parameters = sum(p.numel() for p in model.stgm.bottleneck.parameters())
print(f"compvit: {compvit_total_p:,}")
print(f"compvit bottleneck: {bottleneck_parameters:,}")


print(f"diff: {compvit_total_p - vit_total_p:,}")

