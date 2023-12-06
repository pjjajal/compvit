# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import torch
from lightning.fabric import Fabric
from timm import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEmaV2, NativeScaler, get_state_dict
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

from data.augment import new_data_aug_generator
from data.imagenet import create_imagenet_dataset

# from models_new import CompressiveTransformerFactory
# from models import CompReggyModels, CompReggyModelsv2, CompReggyModelsv3
from spvit.factory import SPViTFactory

torch.set_float32_matmul_precision("medium")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--bce-loss", action="store_true")
    parser.add_argument("--unscale-lr", action="store_true")

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        choices=SPViTFactory._member_names_,
        required=True,
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument("--window_size", default=4, type=int)
    parser.add_argument("--bottleneck_location", default=6, type=int)
    parser.add_argument("--registers", default=32, type=int)

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
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
        "--train-comp",
        dest="train_comp",
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

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.3,
        metavar="PCT",
        help="Color jitter factor (default: 0.3)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
    parser.set_defaults(train_mode=True)

    parser.add_argument("--ThreeAugment", action="store_true")  # 3augment

    parser.add_argument("--src", action="store_true")  # simple random crop

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # # Distillation parameters
    # parser.add_argument(
    #     "--teacher-model",
    #     default="regnety_160",
    #     type=str,
    #     metavar="MODEL",
    #     help='Name of teacher model to train (default: "regnety_160"',
    # )
    # parser.add_argument("--teacher-path", type=str, default="")
    # parser.add_argument(
    #     "--distillation-type",
    #     default="none",
    #     choices=["none", "soft", "hard"],
    #     type=str,
    #     help="",
    # )
    # parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    # parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument("--cosub", action="store_true")

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--nb-classes", default=1000, type=int, help="number of classes"
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    return parser


class TrainEval:
    def __init__(self, args) -> None:
        print(args)

        self.args = args
        self.fabric = Fabric(precision="bf16-mixed")
        self.fabric.launch()

        self.data_loader_train, self.data_loader_val = self._create_datasets()
        self.mixup_fn = self._create_mixup(args)
        self.model: torch.nn.Module = self._create_model(args)
        self.model = self.fabric.setup(self.model)
        self._create_ema(args)
        self.optimizer = self._create_optimizer(args)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.lr_scheduler, _ = create_scheduler(args, self.optimizer)
        self.criterion = self._create_criterion(args)
        self.data_loader_train, self.data_loader_val = self.fabric.setup_dataloaders(
            self.data_loader_train, self.data_loader_val
        )

    def _create_datasets(self):
        args = self.args

        self.train_dataset = create_imagenet_dataset("train", cache_dir=args.data_path)
        self.val_dataset = create_imagenet_dataset("val", cache_dir=args.data_path)

        sampler_train = torch.utils.data.RandomSampler(self.train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(self.val_dataset)

        data_loader_train = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if args.ThreeAugment:
            data_loader_train.dataset.transform = new_data_aug_generator(args)

        data_loader_val = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        return data_loader_train, data_loader_val

    def _create_mixup(self, args):
        mixup_fn = None
        self.mixup_active = (
            args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        )
        if self.mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )

        return mixup_fn

    def _create_model(self, args):
        args = self.args

        # Create model
        model_constructor, model_args = SPViTFactory[args.model].value
        model_args["window_size"] = args.window_size
        model_args["bottleneck_location"] = args.bottleneck_location
        model_args["drop_rate"] = args.drop
        model_args["drop_path_rate"] = args.drop_path
        # model_args["drop_block_rate"] = None
        model_args["img_size"] = args.input_size
        model_args["registers"] = args.registers
        model = model_constructor(**model_args)

        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint), strict=False)

        return model

    def _create_ema(self, args):
        self.model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEmaV2(
                self.model,
                decay=args.model_ema_decay,
                device="cpu" if args.model_ema_force_cpu else self.fabric.device,
            )

    def _create_optimizer(self, args):
        args = self.args
        # Create optimizer
        parameters = []
        if args.whole:
            parameters = self.model.parameters()
        else:
            parameters = self.model.peft_parameters(
                comp_blocks=args.train_comp, blocks=args.blocks
            )

        if args.head:
            parameters.extend(self.model.head.parameters())

        parameters = torch.nn.ParameterList(parameters)
        return create_optimizer(args, parameters)

    def _create_criterion(self, args):
        criterion = LabelSmoothingCrossEntropy()

        if self.mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        if args.bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()

        return criterion

    def evaluate(self):
        loss_mean = MeanMetric().to(device=self.fabric.device)
        accuracy = Accuracy("multiclass", num_classes=1000).to(
            device=self.fabric.device
        )
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for images, target in tqdm(self.data_loader_val):
                output = self.model(images)
                loss = criterion(output, target)
                accuracy(output, target)
                loss_mean(loss)
            total_accuracy = accuracy.compute()
            mean_loss = loss_mean.compute()

        return total_accuracy, mean_loss

    def train(self):
        args = self.args
        logging.info(f"{args}")
        val_acc, val_loss = self.evaluate()
        print(f"Validation Accuracy: {val_acc:.2f}")
        logging.info(f"validation accuracy: {val_acc:.2f}, val loss: {val_loss}")
        for epoch in tqdm(range(args.epochs)):
            losses = self.train_epoch(args)
            self.lr_scheduler.step(epoch)
            logging.info(
                f"Epoch {epoch}: {' '.join([f'{key}: {val}' for key, val in losses.items()])}"
            )
            val_acc, val_loss = self.evaluate()
            logging.info(
                f"epoch: {epoch}, validation accuracy: {val_acc:.2f}, val loss: {val_loss}"
            )

            checkpoint_path = args.save_loc / (
                f"epoch{epoch}" + f"ws{args.window_size}bl{args.bottleneck_location}.pt"
            )
            torch.save(self.model.state_dict(), checkpoint_path)

    def train_epoch(self, args):
        loss_mean = MeanMetric().to(device=self.fabric.device)

        self.model.train()

        criterion = self.criterion
        if args.cosub:
            criterion = torch.nn.BCEWithLogitsLoss()

        for samples, targets in tqdm(self.data_loader_train):
            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)

            if args.cosub:
                samples = torch.cat((samples, samples), dim=0)

            if args.bce_loss:
                targets = targets.gt(0.0).type(targets.dtype)

            outputs = self.model(samples)
            if not args.cosub:
                loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(
                    outputs[0], outputs[1].detach().sigmoid()
                )
                loss = loss + 0.25 * criterion(
                    outputs[1], outputs[0].detach().sigmoid()
                )

            self.fabric.backward(loss)
            self.optimizer.step()

            loss_mean(loss)

            if self.model_ema is not None:
                self.model_ema.update(self.model)

        losses = {
            "batch_loss": loss_mean.compute(),
        }
        return losses


def main():
    CHECKPOINT_PATH = Path("./checkpoints")

    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # Setup Logging
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINT_PATH / now
    if not save_loc.exists():
        save_loc.mkdir()

    log_filename = (
        save_loc
        / f"{now}_{args.model}-ws{args.window_size}bl{args.bottleneck_location}.log"
    )
    logging.basicConfig(
        filename=str(log_filename),
        filemode="w",
        level=logging.INFO,
    )

    args.save_loc = save_loc
    train_eval = TrainEval(args)
    if args.eval:
        val_acc, val_loss = train_eval.evaluate()
        logging.info(f"validation accuracy: {val_acc:.2f}, val loss: {val_loss}")

    else:
        train_eval.train()


if __name__ == "__main__":
    main()
