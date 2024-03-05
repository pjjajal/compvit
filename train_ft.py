import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Union

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
from ffcv.loader import Loader, OrderOption
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torchmetrics import Accuracy

from compvit.factory import compvit_factory
from compvit.models.compvit import CompViT
from datasets import create_dataset
from datasets.imagenet_ffcv import create_train_pipeline, create_val_pipeline
from dinov2.factory import dinov2_factory
from dinov2.models.vision_transformer import DinoVisionTransformer
from utils.schedulers import CosineAnnealingWithWarmup

CONFIG_PATH = Path("./configs")
DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"]
    )
    parser.add_argument("--model", required=True, choices=["dinov2", "compvit"])
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--eval", action="store_true")
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


class LinearClassifierModel(nn.Module):
    def __init__(
        self,
        model: Union[DinoVisionTransformer, CompViT],
        num_classes,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model = model
        self.head = nn.Linear(model.embed_dim * 2, num_classes)

    def forward(self, x):
        outputs = self.model(x, is_training=True)
        cls_token = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]
        outputs = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),
            ],
            dim=1,
        )
        outputs = self.head(outputs)
        return outputs


class LightningFT(L.LightningModule):
    def __init__(
        self,
        model: Union[LinearClassifierModel],
        args,
        hyperparameters,
    ) -> None:
        super().__init__()
        self.model = model
        self.hyperparameters = hyperparameters
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        self.running_loss = 0
        self.highest_val_accuracy = float("-inf")
        self.accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )

    def training_step(self, batch, batch_idx):
        x, label = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, label)
        # Running loss.
        self.running_loss += loss.detach().item()
        self.log("train loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, label)

        # Accuracy
        self.accuracy_top1(outputs, label)
        self.accuracy_top5(outputs, label)
        self.log(
            "accuracy_top1",
            self.accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "accuracy_top5",
            self.accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("test loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.args.model == "dinov2":
            parameters = self.model.head.parameters()
        else:
            parameters = self.model.parameters()
        optimizer = optim.AdamW(
            parameters,
            lr=self.hyperparameters["lr"],
            weight_decay=5e-2,
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

    def on_validation_epoch_end(self) -> None:
        acc = self.accuracy_top1.compute()
        if acc > self.highest_val_accuracy:
            self.highest_val_accuracy = acc
            # Save Model
            save_path = self.args.save_loc / f"best_performing.pth"
            torch.save(self.model.state_dict(), save_path)


def main(args):
    config_path = CONFIG_PATH / (args.dataset + "_dino" + ".yaml")
    configs = OmegaConf.load(config_path)
    model_config = configs[args.model]
    head_config = configs["head"]
    hyperparameters = configs["hyperparameters"]

    # Merging config with CLI args. CLI is prioritized over config.
    args = OmegaConf.merge(
        configs["args"],
        vars(args),
    )

    # Create MAE.
    if args.model == "dinov2":
        model, config = dinov2_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading weights")
            model.load_state_dict(torch.load(model_config["checkpoint"]))

    if args.model == "compvit":
        model, config = compvit_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading", model_config["checkpoint"])
            model.load_state_dict(torch.load(model_config["checkpoint"]), strict=False)

    # Create classifier model.
    model = LinearClassifierModel(model, head_config["num_classes"])

    model = LightningFT(model, args, hyperparameters)

    # Setup W&B.
    wandb_logger = WandbLogger(project="compvit-dt")
    wandb_logger.experiment.config.update(
        {
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
        logger=wandb_logger,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
    )

    # Create dataset and train loader.
    image_pipeline, label_pipeline = create_train_pipeline(
        device=torch.device(f"cuda:{trainer.local_rank}"),
        pretraining=False,
        input_size=224,
        mixup_alpha=hyperparameters["mixup_alpha"],
        num_classes=head_config["num_classes"],
    )
    order = OrderOption.QUASI_RANDOM
    loader = Loader(
        fname=args.data_dir_train,
        batch_size=hyperparameters["batch_size"],
        num_workers=hyperparameters["num_workers"],
        order=order,
        os_cache=hyperparameters["in_memory"],
        drop_last=True,
        pipelines={"image": image_pipeline, "label": label_pipeline},
    )

    # Create test loader.
    test_image_pipeline, test_label_pipeline = create_val_pipeline(
        device=torch.device(f"cuda:{trainer.local_rank}"),
        input_size=224,
    )
    order = OrderOption.SEQUENTIAL
    test_loader = Loader(
        fname=args.data_dir_test,
        batch_size=hyperparameters["batch_size"],
        num_workers=hyperparameters["num_workers"],
        order=order,
        os_cache=hyperparameters["in_memory"],
        drop_last=False,
        pipelines={"image": test_image_pipeline, "label": test_label_pipeline},
    )
    # Trainer Fit.
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    args = parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = DEFAULT_CHECKPOINTS_PATH / now
    if args.checkpoints_path:
        save_loc = args.checkpoints_path / now

    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
