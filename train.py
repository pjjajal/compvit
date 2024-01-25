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

import wandb
from compvit.factory import mae_factory
from datasets import create_dataset
from dinov2.factory import dinov2_factory


torch.set_float32_matmul_precision("medium")

CONFIG_PATH = Path("./configs")
CHECKPOINTS_PATH = Path("./checkpoints")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"]
    )
    parser.add_argument("--downsize", required=True, type=int, default=224)
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

    return parser.parse_args()


def main(args):
    config_path = CONFIG_PATH / (args.dataset + "_pt_dino" + ".yaml")
    configs = OmegaConf.load(config_path)
    baseline_config = configs["teacher"]
    compvit_config = configs["student"]
    hyperparameters = configs["hyperparameters"]

    # Merging config with CLI args. CLI is prioritized over config.
    args = OmegaConf.merge(
        configs["args"],
        vars(args),
    )

    # Initialize and launch fabric.
    fabric = Fabric(
        accelerator="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
    )
    fabric.launch()

    # Setup W&B.
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="compvit-rcac",
        # track hyperparameters and run metadata
        config={
            "architecture": "mae",
            # "dataset": args.dataset,
            "teacher": baseline_config["name"],
            "student": compvit_config["name"],
            **hyperparameters,
            **args,
        },
    )

    # Create dataset and train loader.
    train_dataset, _ = create_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=hyperparameters["num_workers"],
        pin_memory=True
    )

    # Setup dataloader on Fabric.
    train_loader = fabric.setup_dataloaders(train_loader)

    # Get checkpoint paths.
    baseline_checkpoint = baseline_config.pop("checkpoint")
    student_checkpoint = compvit_config.pop("checkpoint")
    decoder_checkpoint = compvit_config.pop("decoder_checkpoint")

    # Create MAE.
    mae, config = mae_factory(
        teacher_name=baseline_config["name"], student_name=compvit_config["name"]
    )
    mae.baseline.load_state_dict(torch.load(baseline_checkpoint))
    mae.encoder.load_state_dict(torch.load(student_checkpoint), strict=False)
    if decoder_checkpoint:
        mae.decoder.load_state_dict(torch.load(decoder_checkpoint))

    # Update W&B run metadata.
    run.config.update(**config)

    # Create optimizer and scheduler.
    optimizer = optim.AdamW(
        mae.training_parameters(whole=True), lr=hyperparameters["lr"], weight_decay=5e-2
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=hyperparameters["warmup_lr_scale"],
        end_factor=1.0,
        total_iters=hyperparameters["warmup_epochs"],
        verbose=True,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        hyperparameters["epochs"] - hyperparameters["warmup_epochs"],
        hyperparameters["min_lr"],
        verbose=True,
    )

    # Setup model and optimizer.
    mae, optimizer = fabric.setup(mae, optimizer)

    # Setup view transform.
    downsize = tvt.Resize(args.downsize)

    lowest_batch_loss = 1e6
    for epoch in range(hyperparameters["epochs"]):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            is_accumulating = (i + 1) % hyperparameters["accumulations"] != 0
            with fabric.no_backward_sync(mae, enabled=is_accumulating):
                # Forward pass.
                loss = mae(img, downsize(img))
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
        wandb.log({"train loss": batch_loss, "lr": optimizer.param_groups[0]["lr"]})

        # Scheduler Step
        if epoch >= hyperparameters["warmup_epochs"]:
            cosine_scheduler.step()
        else:
            warmup_scheduler.step()

        # Save Model
        if lowest_batch_loss > batch_loss:
            lowest_batch_loss = batch_loss
            save_path = args.save_loc / f"best_performing.pth"
            save_path_pt = args.save_loc / f"best_performing_pt.pth"
            torch.save(mae.encoder.state_dict(), save_path)
            torch.save(mae.state_dict(), save_path_pt)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    now = "mae_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
