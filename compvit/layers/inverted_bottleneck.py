import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from functools import partial


def inverted_conv_bottleneck(
    num_tokens,
    code_book_tokens,
    dim,
    ratio,
    inverted_bottleneck_size,
):
    class Permute(nn.Module):
        def __init__(self, dims) -> None:
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return x.permute(self.dims)

    class NeXtTransposeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=1,
                    groups=dim,
                    stride=2,
                ),
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
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=7,
                    padding=3,
                    stride=2,
                    groups=dim
                ),
            )

        def forward(self, x):
            return x + self.blocks(x)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                *[NeXtTransposeBlock() for _ in range(inverted_bottleneck_size)]
            )
            self.pooling = nn.AdaptiveMaxPool1d(
                code_book_tokens if num_tokens < code_book_tokens else num_tokens
            )  # subtract 1 to account for CLS

        def forward(self, x):
            B, N, C = x.shape
            H = W = int(N**0.5)
            cls_token, x = x[:, 0:1, :], x[:, 1:, :]
            x = x.mT.reshape((B, C, H, W))
            x = self.blocks(x)
            x = x.reshape((B, C, -1))
            x = self.pooling(x).mT
            x = torch.concat([cls_token, x], dim=1)
            return x

    return Net()


if __name__ == "__main__":
    x = torch.randn(1, 257, 786).to("cuda")

    t1 = benchmark.Timer(
        stmt="net(x)",
        setup="from __main__ import inverted_conv_bottleneck; net = inverted_conv_bottleneck(256, 256, 786, 4, 2).to('cuda')",
        globals={"x": x},
        num_threads=1,
    )
    print(t1.timeit(100).median)
