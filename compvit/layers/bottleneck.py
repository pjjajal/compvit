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
        nn.BatchNorm1d(num_tokens * ratio),
        nn.GELU(),
        nn.Conv1d(num_tokens * ratio, num_tokens * ratio, 1),
        nn.BatchNorm1d(num_tokens * ratio),
        nn.GELU(),
        nn.Conv1d(num_tokens * ratio, num_compressed_tokens, 1),
        nn.BatchNorm1d(num_compressed_tokens),
        nn.GELU(),
        MixerBlock(dim, num_compressed_tokens),
    )


def mixer_bottleneck_multi_v2(
    num_tokens, num_compressed_tokens, dim, ratio, bottleneck_size
):
    class BottleneckBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(num_tokens * ratio, num_tokens * ratio, 1),
                nn.LayerNorm(dim),
                nn.GELU(),
            )

        def forward(self, x):
            return x + self.block(x)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer_1 = MixerBlock(dim, num_tokens)
            # Non-linear projection from num_tokens -> num_tokens * ratio
            self.up_block = nn.Sequential(
                nn.Conv1d(num_tokens, num_tokens * ratio, 1),
                nn.LayerNorm(dim),
                nn.GELU(),
            )

            self.bottleneck_blocks = nn.Sequential(
                *[BottleneckBlock() for i in range(bottleneck_size)]
            )

            # Non-linear projection from num_tokens * ratio -> num_compressed_tokens
            self.down_block = nn.Sequential(
                nn.Conv1d(num_tokens * ratio, num_compressed_tokens, 1),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            self.mixer_2 = MixerBlock(dim, num_compressed_tokens)

        def forward(self, x):
            x = self.mixer_1(x)
            x = self.up_block(x)
            x = self.bottleneck_blocks(x)
            x = self.down_block(x)
            x = self.mixer_2(x)
            return x

    return Net()
