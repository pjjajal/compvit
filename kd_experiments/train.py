import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.imagenet import create_imagenet_dataset
from .kd_model import FeatureKD

# import wandb

torch.set_float32_matmul_precision("medium")

CHECKPOINTS_PATH = Path("./checkpoints")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument("--data_dir", type=Path)

    # Model Setup
    parser.add_argument("--student", choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    parser.add_argument("--teacher", choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    parser.add_argument("--loss", choices=["l1-smooth", "l2"])
    
    # Training related
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--warmup_lr_scale", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accumulations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=[
            "32-true",
            "32",
            "16-mixed",
            "bf16-mixed",
            "transformer-engine",
            "16-true",
            "bf16-true",
            "64-true",
        ],
        default="bf16-mixed",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Initialize and launch fabric.
    fabric = Fabric(
        accelerator="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
    )
    fabric.launch()

    # Create dataset and train loader.
    train_dataset = create_imagenet_dataset(split="train", data_dir=args.data_dir, pretraining=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Setup dataloader on Fabric.
    train_loader = fabric.setup_dataloaders(train_loader)

    model = FeatureKD(student_name=args.student, teacher_name=args.teacher, loss=args.loss)

    optimizer = optim.AdamW(
        model.training_parameters(), lr=args.lr, weight_decay=5e-2
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_lr_scale,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
        verbose=True,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.epochs - args.warmup_epochs,
        args.min_lr,
        verbose=True,
    )

    model, optimizer = fabric.setup(model, optimizer)

    lowest_batch_loss = 1e6
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            is_accumulating = (i + 1) % args.accumulations != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # Forward pass.
                loss = model(img, img)
                # Running loss.
                running_loss += loss.detach().item()
                # Backward pass.
                fabric.backward(loss)
            if not is_accumulating or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        batch_loss = running_loss / len(train_loader)

        # Logging
        print(
            f"batch loss: {batch_loss}, , lr: {optimizer.param_groups[0]['lr']}"
        )
        # wandb.log({"train loss": batch_loss, "lr": optimizer.param_groups[0]["lr"]})

        # Scheduler Step
        if epoch >= args.warmup_epochs:
            cosine_scheduler.step()
        else:
            warmup_scheduler.step()

        # Save Model
        if lowest_batch_loss > batch_loss:
            lowest_batch_loss = batch_loss
            save_path = args.save_loc / f"best_performing.pth"
            save_path_pt = args.save_loc / f"best_performing_kd.pth"
            torch.save(model.student.state_dict(), save_path)
            torch.save(model.state_dict(), save_path_pt)
    # wandb.finish()

if __name__ == "__main__":
    args = parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
