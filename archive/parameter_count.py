from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory


TOY_EXPERIMENTS_PATH = Path("./toy_experiments")
CONFIG_PATH = TOY_EXPERIMENTS_PATH / "configs"

config_path = CONFIG_PATH / ("cifar100" + ".yaml")

model, _ = dinov2_factory("dinov2_vits14")

vit_total_p = sum(p.numel() for p in model.parameters())
print(f"vit:  {vit_total_p:,}")

model, _ = compvit_factory("compvits14")
compvit_total_p = sum(p.numel() for p in model.parameters())
bottleneck_parameters = sum(p.numel() for p in model.compressor.bottleneck.parameters())
print(f"compvit: {compvit_total_p:,}")
print(f"compvit bottleneck: {bottleneck_parameters:,}")


print(f"diff: {compvit_total_p - vit_total_p:,}")

