import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mlp_mixer import MixerBlock
import torch.utils.benchmark as benchmark


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


def conv_bottleneck(num_tokens, num_compressed_tokens, dim, ratio, bottleneck_size):
    class Permute(nn.Module):
        def __init__(self, dims) -> None:
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return x.permute(self.dims)

    class NeXtBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=7,
                    padding="same",
                    groups=dim,
                ),
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(dim),
                nn.Linear(in_features=dim, out_features=dim * ratio),
                nn.GELU(),
                nn.Linear(in_features=dim * ratio, out_features=dim),
                Permute((0, 3, 1, 2)),
            )

        def forward(self, x):
            return x + self.blocks(x)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(*[NeXtBlock() for _ in range(bottleneck_size)])
            self.pooling = nn.AdaptiveAvgPool2d(int(num_compressed_tokens**0.5))

        def forward(self, x):
            B, N, C = x.shape
            H = W = int(N**0.5)

            x = x.mT.reshape((B, C, H, W))
            x = self.blocks(x)
            x = self.pooling(x)
            x = x.reshape((B, C, num_compressed_tokens))
            return x

    return Net()


if __name__ == "__main__":
    net = conv_bottleneck(256, 16, 786, 1, 1)
    print(net(torch.randn(32, 256, 786)).shape)

    x = torch.randn(32, 256, 786).to('cuda')
    t0 = benchmark.Timer(
        stmt="net(x)",
        setup="from __main__ import conv_bottleneck; net = conv_bottleneck(256, 16, 786, 4, 1).to('cuda')",
        globals={"x": x},
        num_threads=1
    )

    t1 = benchmark.Timer(
        stmt="net(x)",
        setup="from __main__ import mixer_bottleneck_multi_v2; net = mixer_bottleneck_multi_v2(256, 16, 786, 4, 1).to('cuda')",
        globals={"x": x},
        num_threads=1
    )
    print(t0.timeit(100).median)
    print(t1.timeit(100).median)