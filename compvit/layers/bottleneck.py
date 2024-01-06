import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mlp_mixer import MixerBlock


def mixer_bottleneck(num_tokens, num_compressed_tokens, dim):
    return nn.Sequential(
        MixerBlock(dim, num_tokens),
        nn.Conv1d(num_tokens, num_compressed_tokens, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.GELU(),
        MixerBlock(dim, num_compressed_tokens),
    )


def mixer_bottleneck_relu(num_tokens, num_compressed_tokens, dim):
    return nn.Sequential(
        MixerBlock(dim, num_tokens),
        nn.Conv1d(num_tokens, num_compressed_tokens, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.ReLU(),
        MixerBlock(dim, num_compressed_tokens),
    )


def mixer_bottleneck_multi(num_tokens, num_compressed_tokens, dim, ratio):
    return nn.Sequential(
        MixerBlock(dim, num_tokens),
        nn.Conv1d(num_tokens, num_tokens * ratio, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.GELU(),
        nn.Conv1d(num_tokens * ratio, num_tokens * ratio, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.GELU(),
        nn.Conv1d(num_tokens * ratio, num_compressed_tokens, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.GELU(),
        MixerBlock(dim, num_compressed_tokens),
    )
