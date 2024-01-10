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
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from tqdm import tqdm

import wandb
from compvit.factory import compvit_factory, mae_factory
from dinov2.factory import dinov2_factory
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

TOY_EXPERIMENTS_PATH = Path("./toy_experiments")
DATA_PATH = TOY_EXPERIMENTS_PATH / "data"
CONFIG_PATH = TOY_EXPERIMENTS_PATH / "configs"
CHECKPOINTS_PATH = TOY_EXPERIMENTS_PATH / "checkpoints_dino"


def create_transform(args):
    if args.dataset in ["cifar10", "cifar100"]:
        return tvt.Compose(
            [
                tvt.RandomCrop(32, padding=4),
                tvt.Resize(224),
                tvt.RandomHorizontalFlip(),
                tvt.ToTensor(),
                tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    if args.dataset == "imagenet":
        return tvt.Compose(
            [
                    tvt.RandomResizedCrop(
                        args.input_size, scale=(0.2, 1.0), interpolation=3
                    ),  # 3 is bicubic
                    tvt.RandomHorizontalFlip(),
                    tvt.ToTensor(),
                    tvt.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--downsize", required=True, type=int, default=224)

    return parser.parse_args()


def create_dataset(args, DATA_PATH):
    train_dataset = None
    TRANSFORM = create_transform(args)
    if args.dataset == "cifar10":
        train_dataset = CIFAR10(DATA_PATH, transform=TRANSFORM, download=True)
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(DATA_PATH, transform=TRANSFORM, download=True)
    elif args.dataset == "imagenet":
        train_dataset = ImageNet(data_loc, transform=TRANSFORM)
    return train_dataset


def main(args):
    config_path = CONFIG_PATH / (args.dataset + "_mae_dino" + ".yaml")
    configs = OmegaConf.load(config_path)
    baseline_config = configs["teacher"]
    compvit_config = configs["student"]
    hyperparameters = configs["hyperparameters"]
    device = configs["device"]
    DATA_PATH = getattr(configs, 'data_loc', DATA_PATH)

    downsize = tvt.Resize(args.downsize)

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="compvit-rcac",
        # track hyperparameters and run metadata
        config={
            "architecture": "mae",
            "dataset": args.dataset,
            "teacher": baseline_config["name"],
            "student": compvit_config["name"],
            **hyperparameters,
        },
    )

    train_dataset = create_dataset(args, DATA_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=8,
    )

    baseline_checkpoint = baseline_config.pop("checkpoint")
    student_checkpoint = compvit_config.pop("checkpoint")
    decoder_checkpoint = compvit_config.pop("decoder_checkpoint")

    mae, config = mae_factory(
        teacher_name=baseline_config["name"], student_name=compvit_config["name"]
    )
    mae.baseline.load_state_dict(torch.load(baseline_checkpoint))
    mae.encoder.load_state_dict(torch.load(student_checkpoint), strict=False)
    if decoder_checkpoint:
        mae.decoder.load_state_dict(torch.load(decoder_checkpoint))
    mae = mae.to(device=device)

    run.config.update(config)

    optimizer = optim.AdamW(
        mae.training_parameters(whole=True), lr=hyperparameters["lr"], weight_decay=5e-2
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=hyperparameters["warmup_lr_scale"],
        end_factor=1.0,
        total_iters=hyperparameters["warmup_epochs"],
        verbose=True
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        hyperparameters["epochs"] - hyperparameters["warmup_epochs"],
        hyperparameters["min_lr"],
        verbose=True
    )
    

    # scaler = torch.cuda.amp.grad_scaler.GradScaler()
    lowest_batch_loss = 1e6
    for epoch in range(hyperparameters["epochs"]):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device="cuda")
            label = label.to(device="cuda")

            # optimizer.zero_grad()
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, other_loss = mae(img, downsize(img))
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            is_accumulating = (i + 1) % hyperparameters['accumulations'] != 0
            # loss = loss / hyperparameters['accumulations']
            running_loss += loss.detach().item()
            loss.backward()
            if not is_accumulating or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        batch_loss = running_loss / len(train_loader)

        # Logging
        print(f"batch loss: {batch_loss}, , lr: {optimizer.param_groups[0]['lr']}, {other_loss}")
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
            save_path_mae = args.save_loc / f"best_performing_mae.pt"
            torch.save(mae.encoder.state_dict(), save_path)
            torch.save(mae.state_dict(), save_path_mae)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    now = "mae_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    args.save_loc = save_loc
    main(args)
