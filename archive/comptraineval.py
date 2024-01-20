import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from timm.models.vision_transformer import Mlp
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

from spvit.factory import SPViTFactory


torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computes the inference time for model with a sweep of registers and pruning schedules."
    )
    parser.add_argument(
        "--model", type=str, choices=SPViTFactory._member_names_, required=True
    )
    parser.add_argument("--registers", type=int, required=True)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-min",
        default=1e-4,
        type=float,
        metavar="LR",
        help="final learning rate",
        dest="lr_min",
    )
    parser.add_argument(
        "--tmax",
        default=10,
        type=int,
        help="T max for cosine annealing",
        dest="tmax",
    )
    parser.add_argument(
        "--tmult",
        default=1,
        type=int,
        help="T mult for cosine annealing",
        dest="tmult",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint name",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--whole",
        dest="whole",
        action="store_true",
        help="evaluate model on validation set",
        default=False,
    )
    parser.add_argument(
        "--blocks",
        dest="blocks",
        action="store_true",
        help="evaluate model on validation set",
        default=False,
    )
    parser.add_argument(
        "--train-reg",
        dest="train_reg",
        action="store_true",
        help="evaluate model on validation set",
        default=False,
    )
    parser.add_argument(
        "--head",
        dest="head",
        action="store_true",
        help="evaluate model on validation set",
        default=False,
    )
    parser.add_argument(
        "--save", dest="save", help="save_name", type=Path, default=None
    )
    args = parser.parse_args()
    return args


class TrainEval:
    def __init__(self, args) -> None:
        self.args = args

        self.logger = TensorBoardLogger(root_dir="logs")
        self.fabric = Fabric(loggers=self.logger, precision="bf16-mixed")
        self.fabric.launch()

        self._create_datasets()
        self._create_model()
        self._create_mixup()
        self.model = self.fabric.setup(self.model)
        print(self.model)
        self._setup_model_optimizers()
        self.optimizer = self.fabric.setup_optimizers(*self.optimizers)
        # self.warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, 1/50, 1, 5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=args.tmax, eta_min=args.lr_min, T_mult=args.tmult
        )
        self.train_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.val_dataloader
        )

    def _create_datasets(self):
        args = self.args

        # Datasets
        train_sampler = None
        val_sampler = None

        self.train_dataset = create_imagenet_dataset("train", cache_dir=args.cache_dir)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        self.val_dataset = create_imagenet_dataset("val", cache_dir=args.cache_dir)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
        )

    def _create_mixup(self):
        args = OmegaConf.create(
            {
                "mixup": 0.8, # This is set in the CLI of DEIT
                "cutmix": 1.0, # This is set in the CLI of DEIT
                "cutmix_minmax": None, # This is the default value from DEIT
                "mixup_prob": 1.0,
                "mixup_switch_prob": 0.5,
                "mixup_mode": "batch", 
                "smoothing": 0.0, # This is set in the CLI of DEIT
                "nb_classes": 1000 # This is set in CLI of DEIT
            }
        )
        self.mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    def _create_model(self):
        args = self.args

        # Create model
        model_constructor, model_args = CompReggyModelsv3[args.model].value
        model_args["registers"] = args.registers
        self.model = model_constructor(**model_args)

        if args.checkpoint:
            self.model.load_state_dict(torch.load(args.checkpoint), strict=False)

    def _setup_model_optimizers(self):
        args = self.args
        self.optimizers = []

        # Create optimizer
        parameters = []
        if args.whole:
            parameters = self.model.parameters()
        else:
            parameters = self.model.peft_parameters(
                registers=args.train_reg, blocks=args.blocks
            )

        if args.head:
            parameters.extend(self.model.head.parameters())

        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=0.05
        )
        # optimizer = torch.optim.SGD(
        #     parameters, lr=args.lr, weight_decay=1e-4, momentum=0.9, nesterov=True
        # )
        self.optimizers.append(optimizer)

    def validate(self):
        loss_mean = MeanMetric().to(device=self.fabric.device)
        accuracy = Accuracy("multiclass", num_classes=1000).to(
            device=self.fabric.device
        )
        ce = nn.CrossEntropyLoss().to(self.fabric.device)

        self.model.eval()

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_dataloader):
                pred, _ = self.model(imgs)
                loss = ce(pred, labels)
                accuracy(pred, labels)
                loss_mean(loss)
            total_accuracy = accuracy.compute()
            mean_loss = loss_mean.compute()
        # print(f"Validation Accuracy: {total_accuracy:.2f}")

        return total_accuracy, mean_loss

    def train(self):
        args = self.args
        logging.info(f"LR: {args.lr}; LR_MIN: {args.lr_min}; T_max: {args.tmax}")
        val_acc, val_loss = self.validate()
        print(f"Validation Accuracy: {val_acc:.2f}")
        logging.info(f"validation accuracy: {val_acc:.2f}, val loss: {val_loss}")
        for epoch in tqdm(range(args.epochs)):
            losses = self.train_epoch()
            # if epoch < 5:
            #     self.warmup.step()
            #     print(self.warmup.get_last_lr())
            # else:
            self.scheduler.step()
            print(self.scheduler.get_last_lr())
            self.fabric.log_dict({"epoch": epoch, **losses}, step=epoch + 1)
            logging.info(
                f"Epoch {epoch}: {' '.join([f'{key}: {val}' for key, val in losses.items()])}"
            )

            if epoch % 1 == 0:
                # Training r validation
                val_acc, val_loss = self.validate()
                # self.fabric.log("validation accuracy", val_acc, step=epoch + 1)
                print(f"epoch: {epoch}, validation accuracy: {val_acc:.2f}")
                logging.info(
                    f"epoch: {epoch}, validation accuracy: {val_acc:.2f}, val loss: {val_loss}"
                )
                ########
                torch.save(
                    self.model.state_dict(),
                    f"epoch{epoch}" + f"reg{args.registers}.pt",
                )

            if args.save:
                torch.save(self.model.state_dict(), args.save)

    def train_epoch(self):
        loss_mean = MeanMetric().to(device=self.fabric.device)
        ce_mean = MeanMetric().to(device=self.fabric.device)
        tce_mean = MeanMetric().to(device=self.fabric.device)

        ce = nn.CrossEntropyLoss().to(self.fabric.device)

        i = 0
        self.model.train()
        for imgs, labels in tqdm(self.train_dataloader):
            i += 1
            imgs, labels = self.mixup_fn(imgs, labels)
            pred, s_int_features = self.model(imgs)

            ce_loss = ce(pred, labels)
            loss = 1 * ce_loss

            self.fabric.backward(loss)
            self.optimizer.step()
            # self.fabric.log_dict(
            #     {
            #         "batch_loss": loss,
            #         "ce_loss": ce_loss,
            #     },
            #     step=i,
            # )

            ce_mean(ce_loss)
            loss_mean(loss)

        losses = {
            "batch_loss": loss_mean.compute(),
            "ce_loss": ce_mean.compute(),
        }
        return losses


def main():
    args = parse_args()
    # Setup Logging
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    logging.basicConfig(
        filename=f"{now}_{args.model}-reg{args.registers}.log",
        filemode="w",
        level=logging.INFO,
    )

    train_eval = TrainEval(args)
    if args.evaluate:
        train_eval.validate()
    else:
        train_eval.train()


if __name__ == "__main__":
    main()
