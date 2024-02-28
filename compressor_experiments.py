# %%
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from tqdm import tqdm

from compvit.factory import compvit_factory
from compvit.layers.bottleneck import conv_bottleneck
from compvit.layers.compressor import Compressor
from compvit.layers.inverted_bottleneck import inverted_mlp
from dinov2.factory import dinov2_factory
from dinov2.layers import Mlp

# %%
embed_dim = 384
num_heads = 6
mlp_ratio = 4
qkv_bias = True
ffn_bias = True
proj_bias = True
norm_layer = partial(nn.LayerNorm, eps=1e-6)
act_layer = nn.GELU
ffn_layer = Mlp
init_values = 1.0
num_compressed_tokens = 17
total_tokens = 257

bottleneck_size = 1
bottleneck = partial(
    conv_bottleneck,
    dim=embed_dim,
    ratio=mlp_ratio,
    bottleneck_size=bottleneck_size,
)

inv_bottle_size = 1
codebook_ratio = 2
inv_bottleneck = partial(
    inverted_mlp,
    dim=embed_dim,
    ratio=codebook_ratio,
)




def colour_text(
    text,
    colour_code: Literal[
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "reset",
    ],
    *args,
    **kwargs,
):
    colour_codes = {
        "black": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    coloured_text = colour_codes[colour_code] + str(text) + colour_codes["reset"]
    return coloured_text


class CompresserDecoder(nn.Module):
    def __init__(self, compressor, total_tokens, num_compressed_tokens) -> None:
        super().__init__()
        self.compressor = compressor
        self.decoder = nn.Linear(num_compressed_tokens, total_tokens)

    def forward(self, x):
        x = self.compressor(x)
        x = self.decoder(x.mT).mT
        return x


def calculate_mean_std(dataset):
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for img, _ in loader:
        batch_mean = torch.mean(img, dim=(0, 2, 3))
        batch_std = torch.std(img, dim=(0, 2, 3))
        mean += batch_mean
        std += batch_std
    mean /= len(loader)
    std /= len(loader)
    return mean, std


def synthetic_dataset():
    mean, std = calculate_mean_std(
        ImageFolder(root="data/synthetic-checkerboard/train", transform=tvt.ToTensor())
    )
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize(mean, std)])
    train_dataset = ImageFolder(
        "data/synthetic-checkerboard/train", transform=transform
    )
    val_dataset = ImageFolder("data/synthetic-checkerboard/val", transform=transform)
    return train_dataset, val_dataset


def cifar_dataset(type: Literal["cifar10", "cifar100"]):
    if type not in ["cifar10", "cifar100"]:
        raise ValueError("Invalid dataset type")
    TRANSFORM = tvt.Compose(
        [
            tvt.RandomCrop(32, padding=4),
            tvt.Resize(224),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    TRANSFORM_TEST = tvt.Compose(
        [
            tvt.Resize(224),
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if type == "cifar10":
        train_dataset = CIFAR10(
            "toy_experiments/data", transform=TRANSFORM, download=True
        )
        val_dataset = CIFAR10(
            "toy_experiments/data", transform=TRANSFORM_TEST, train=False, download=True
        )
    if type == "cifar100":
        train_dataset = CIFAR100(
            "toy_experiments/data", transform=TRANSFORM, download=True
        )
        val_dataset = CIFAR100(
            "toy_experiments/data", transform=TRANSFORM_TEST, train=False, download=True
        )
    return train_dataset, val_dataset


def evaluate_reconstruction(model, compressor_decoder, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, _ in tqdm(val_loader):
            img = img.to("cuda")
            tokens = model.prepare_tokens_with_masks(img)
            pred_tokens = compressor_decoder(tokens)
            loss = F.mse_loss(pred_tokens, tokens, reduction="mean")
            total_loss += loss.detach().item()
    return total_loss / len(val_loader)


def train(model, compressor_decoder, train_loader, val_loader, epochs, optimizer):
    for epoch in tqdm(range(epochs)):
        compressor_decoder.train()
        total_loss = 0.0
        for img, _ in train_loader:
            img = img.to("cuda")
            with torch.no_grad():
                tokens = model.prepare_tokens_with_masks(img)
            pred_tokens = compressor_decoder(tokens)
            loss = F.mse_loss(pred_tokens, tokens, reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = loss.item()
        # print(f"{epoch + 1}/{epochs} | {colour_text("Train Loss", "yellow")}: {total_loss / len(train_loader):.2e}")
        # print(f'{colour_text("Reconstruction Loss", "yellow")}: {evaluate_reconstruction(model, compressor_decoder, val_loader):.2e}')


def create_compressor(num_codebook_tokens):
    compressor = Compressor(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        ffn_bias=ffn_bias,
        norm_layer=norm_layer,
        act_layer=act_layer,
        ffn_layer=ffn_layer,
        init_values=init_values,
        num_compressed_tokens=num_compressed_tokens,
        num_tokens=total_tokens,
        bottleneck=bottleneck,
        num_codebook_tokens=num_codebook_tokens,
        inv_bottleneck=inv_bottleneck,
    )
    return compressor


def evaluate(
    dataset: Literal["synthetic", "cifar10", "cifar100"], codebook_tokens: list[int]
):
    model, cfg = dinov2_factory("dinov2_vits14")
    model.load_state_dict(torch.load("dinov2/checkpoints/dinov2_vits14_pretrain.pth"))
    model = model.to("cuda")
    model.train()

    if dataset not in ["synthetic", "cifar10", "cifar100"]:
        raise ValueError("Invalid dataset type")
    if dataset == "synthetic":
        train_dataset, val_dataset = synthetic_dataset()
    if dataset == "cifar10":
        train_dataset, val_dataset = cifar_dataset("cifar10")
    if dataset == "cifar100":
        train_dataset, val_dataset = cifar_dataset("cifar100")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)

    all_data = {num_codebook_token: {} for num_codebook_token in codebook_tokens}
    for num_codebook_token in codebook_tokens:
        print(f"Dataset: {dataset}, codebook tokens: {num_codebook_token}")
        compressor = create_compressor(num_codebook_token)
        compressor_decoder = CompresserDecoder(
            compressor,
            total_tokens,
            num_compressed_tokens,
        ).to("cuda")
        total_params = sum(p.numel() for p in compressor_decoder.parameters())
        print(f"Number of parameters in CompressorDecoder: {total_params:,}")

        epochs = 50
        optimizer = torch.optim.AdamW(
            compressor_decoder.parameters(), lr=1e-6, weight_decay=5e-2
        )

        train(model, compressor_decoder, train_loader, val_loader, epochs, optimizer)
        test_loss = evaluate_reconstruction(model, compressor_decoder, val_loader)
        print(f'{colour_text("Reconstruction Loss", "green")}: {test_loss:.2e}')
        all_data[num_codebook_token]["test_loss"] = test_loss

    message = ""
    for num_codebook_token, data in all_data.items():
        message += f'{colour_text("Dataset", "green")}: {dataset}, '
        message += f'{colour_text("Codebook Tokens", "red")}: {num_codebook_token}, '
        message += (
            f'{colour_text("Reconstruction Error", "cyan")}: {data["test_loss"]:.2e}'
        )
        message += "\n"
    print(message)


if __name__ == "__main__":
    # evaluate("synthetic", [384, 257, 128])
    evaluate("cifar10", [384, 257, 128])
    evaluate("cifar100", [384, 257, 128])
