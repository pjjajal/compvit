import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mlp_mixer import MixerBlock


def mixer_bottleneck(num_tokens, num_compressed_tokens, dim):
    return nn.Sequential(
        MixerBlock(dim, num_tokens),
        nn.Conv1d(num_tokens, num_compressed_tokens, 1),
        nn.GELU(),
        MixerBlock(dim, num_compressed_tokens),
    )
