import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tvt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

import wandb
from compvit.factory import compvit_factory, mae_factory
from dinov2.factory import dinov2_factory

TOY_EXPERIMENTS_PATH = Path("./toy_experiments")
DATA_PATH = TOY_EXPERIMENTS_PATH / "data"
CONFIG_PATH = TOY_EXPERIMENTS_PATH / "configs"
CHECKPOINTS_PATH = TOY_EXPERIMENTS_PATH / "checkpoints_dino"


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
        tvt.Resize(32),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    # parser.add_argument("--model", required=True, choices=["dinov2", "compvit"])

    return parser.parse_args()


def create_dataset(args):
    train_dataset = None
    test_dataset = None
    if args.dataset == "cifar10":
        train_dataset = CIFAR10(DATA_PATH, transform=TRANSFORM, download=True)
        test_dataset = CIFAR10(
            DATA_PATH, transform=TRANSFORM_TEST, train=False, download=True
        )
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(DATA_PATH, transform=TRANSFORM, download=True)
        test_dataset = CIFAR100(
            DATA_PATH, transform=TRANSFORM_TEST, train=False, download=True
        )
        pass
    return train_dataset, test_dataset


def main(args):
    config_path = CONFIG_PATH / (args.dataset + "_mae_dino" + ".yaml")
    configs = OmegaConf.load(config_path)
    baseline_config = configs["teacher"]
    compvit_config = configs["student"]
    hyperparameters = configs["hyperparameters"]

    wandb.init(
        # set the wandb project where this run will be logged
        project="compvit-toy",
        # track hyperparameters and run metadata
        config={
            "architecture": "mae",
            "dataset": args.dataset,
            "teacher": baseline_config["name"],
            "student": compvit_config["name"],
            **hyperparameters,
        },
    )

    train_dataset, _ = create_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=8,
    )

    baseline_checkpoint = baseline_config.pop("checkpoint")

    mae = mae_factory(
        teacher_name=baseline_config["name"], student_name=compvit_config["name"]
    )
    mae.baseline.load_state_dict(torch.load(baseline_checkpoint))
    # decoder_layer = nn.TransformerDecoderLayer(
    #     d_model=baseline_config["embed_dim"],
    #     nhead=6,
    #     dim_feedforward=int(baseline_config["embed_dim"] * 4),
    #     dropout=0.0,
    #     activation=F.gelu,
    #     layer_norm_eps=1e-5,
    #     batch_first=True,
    #     norm_first=True,
    # )
    # decoder = nn.TransformerDecoder(decoder_layer, 4)

    # mae = MAEViT(
    #     baseline=baseline,
    #     encoder=model,
    #     decoder=decoder,
    #     baseline_embed_dim=baseline.embed_dim,
    #     embed_dim=model.embed_dim,
    #     decoder_embed_dim=decoder_config["decoder_embed_dim"],
    #     norm_layer=nn.LayerNorm,
    # ).to(device="cuda")

    optimizer = optim.AdamW(
        mae.training_parameters(True), lr=hyperparameters["lr"], weight_decay=0.05
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=hyperparameters["warmup_lr_scale"],
        end_factor=1.0,
        total_iters=hyperparameters["warmup_epochs"],
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        hyperparameters["epochs"] - hyperparameters["warmup_epochs"],
        hyperparameters["min_lr"],
    )

    lowest_batch_loss = 1e6
    for epoch in range(hyperparameters["epochs"]):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device="cuda")
            label = label.to(device="cuda")

            optimizer.zero_grad()

            loss = mae(img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        batch_loss = running_loss / len(train_loader)

        # Logging
        print(f"batch loss: {batch_loss}, , lr: {optimizer.param_groups[0]['lr']}")
        wandb.log({"train loss": batch_loss, "lr": optimizer.param_groups[0]["lr"]})

        # Scheduler Step
        if epoch >= hyperparameters["warmup_epochs"]:
            cosine_scheduler.step()
        else:
            warmup_scheduler.step()

        # Save Model
        if lowest_batch_loss > batch_loss:
            lowest_batch_loss = batch_loss
            save_path = args.save_loc / f"best_performing.pt"
            torch.save(mae.encoder.state_dict(), save_path)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    now = "mae_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    args.save_loc = save_loc
    main(args)
