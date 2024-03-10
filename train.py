import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
import wandb
from ffcv.loader import Loader, OrderOption
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_optimizer.shampoo import Shampoo

from compvit.factory import mae_factory
from compvit.models.mae import MAECompVit
from datasets import create_dataset
from datasets.imagenet_ffcv import create_train_pipeline, create_val_pipeline
from dinov2.factory import dinov2_factory
from utils.schedulers import CosineAnnealingWithWarmup

CONFIG_PATH = Path("./configs")
DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"]
    )
    parser.add_argument("--downsize", required=True, type=int, default=224)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--checkpoints_path", type=Path, default=None)
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


class LightningMAE(L.LightningModule):
    def __init__(self, model: MAECompVit, args, hyperparameters) -> None:
        super().__init__()
        self.model = model
        self.hyperparameters = hyperparameters
        self.args = args
        self.downsize = tvt.Resize(args.downsize)

        self.running_loss = 0
        self.lowest_batch_loss = float("inf")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, loss_dict = self.model(x, self.downsize(x))
        # Running loss.
        self.running_loss += loss.detach().item()
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        self.log(
            "train loss",
            loss.detach().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        #     self.model.training_parameters(whole=True),
        #     # self.model.parameters(),
        #     lr=self.hyperparameters["lr"],
        #     weight_decay=5e-2,
        # )
        optimizer = Shampoo(
            self.model.training_parameters(whole=True),
            lr=self.hyperparameters["lr"],
            weight_decay=3.75e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWithWarmup(
                    optimizer,
                    T_max=self.hyperparameters["epochs"],
                    eta_min=self.hyperparameters["min_lr"],
                    warmup_epochs=self.hyperparameters["warmup_epochs"],
                ),
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self) -> None:
        if self.running_loss < self.lowest_batch_loss:
            self.lowest_batch_loss = self.running_loss
            self.running_loss = 0
            # Save Model
            save_path = self.args.save_loc / f"best_performing.pth"
            save_path_pt = self.args.save_loc / f"best_performing_pt.pth"
            torch.save(self.model.encoder.state_dict(), save_path)
            torch.save(self.model.state_dict(), save_path_pt)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)


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

    # Get checkpoint paths.
    baseline_checkpoint = baseline_config.pop("checkpoint")
    student_checkpoint = compvit_config.pop("checkpoint")
    decoder_checkpoint = compvit_config.pop("decoder_checkpoint")

    # Create MAE.
    model, config = mae_factory(
        teacher_name=baseline_config["name"], student_name=compvit_config["name"]
    )
    if baseline_checkpoint:
        model.baseline.load_state_dict(torch.load(baseline_checkpoint), strict=False)
    if student_checkpoint:
        model.encoder.load_state_dict(torch.load(student_checkpoint), strict=False)
    if decoder_checkpoint:
        model.decoder.load_state_dict(torch.load(decoder_checkpoint))

    model = LightningMAE(model, args, hyperparameters)

    # Setup W&B.
    wandb_logger = WandbLogger(project="compvit-rcac")
    wandb_logger.experiment.config.update(
        {
            "architecture": "mae",
            # "dataset": args.dataset,
            "teacher": baseline_config["name"],
            "student": compvit_config["name"],
            **config,
            **hyperparameters,
            **args,
        }
    )

    # Create lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Create trainer.
    trainer = L.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=hyperparameters["accumulations"],
        max_epochs=hyperparameters["epochs"],
        gradient_clip_val=hyperparameters["gradient_clip_val"],
        logger=wandb_logger,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
    )

    # Create dataset and train loader.
    image_pipeline, label_pipeline = create_train_pipeline(
        device=torch.device(f"cuda:{trainer.local_rank}"),
        pretraining=True,
        input_size=224,
    )
    order = OrderOption.RANDOM if args.devices > 1 else OrderOption.QUASI_RANDOM
    loader = Loader(
        fname=args.data_dir,
        batch_size=hyperparameters["batch_size"],
        num_workers=hyperparameters["num_workers"],
        order=order,
        os_cache=hyperparameters["in_memory"],
        drop_last=True,
        pipelines={"image": image_pipeline, "label": label_pipeline},
    )
    # Trainer Fit.
    trainer.fit(model, loader)


if __name__ == "__main__":
    args = parse_args()

    now = "mae_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = DEFAULT_CHECKPOINTS_PATH / now
    if args.checkpoints_path:
        save_loc = args.checkpoints_path / now

    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
