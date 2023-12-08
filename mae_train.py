import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

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
from data.dataset import create_imagenet_dataset
from mae.spvit_mae import mae_factory, MAEViT

torch.set_float32_matmul_precision("medium")


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_tiny_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--baseline_model",
        default="vit_tiny_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint name",
        type=Path,
        default=None,
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument("--window_size", default=6, type=int)
    parser.add_argument("--bottleneck_location", default=6, type=int)
    parser.add_argument("--stgm_location", default=[5,6],)

    # Parameters to optimize
    parser.add_argument(
        "--whole",
        dest="whole",
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
        default=[0.9, 0.95],
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

    # LR Scheduler Settings
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
        default=40,
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

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


class TrainEval:
    def __init__(self, args) -> None:
        print(args)

        self.args = args
        self.fabric = Fabric(precision="bf16-mixed")
        self.fabric.launch()
        self.data_loader_train, self.data_loader_val = self._create_datasets()
        self.model: MAEViT = self._create_model(args)
        self.model = self.fabric.setup(self.model)

        self.optimizer = self._create_optimizer(args)
        self.lr_scheduler, _ = create_scheduler(args, self.optimizer)

        # self.data_loader_train, self.data_loader_val = self.fabric.setup_dataloaders(
        # self.data_loader_train, self.data_loader_val
        # )

        self.data_loader_train = self.fabric.setup_dataloaders(self.data_loader_train)

    def _create_datasets(self):
        args = self.args

        self.train_dataset = create_imagenet_dataset(
            "train", cache_dir=args.data_path, args=args, mae=True
        )
        # self.val_dataset = create_imagenet_dataset(
        #     "val", cache_dir=args.data_path, args=args
        # )

        sampler_train = torch.utils.data.RandomSampler(self.train_dataset)
        # sampler_val = torch.utils.data.SequentialSampler(self.val_dataset)

        data_loader_train = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        # data_loader_val = torch.utils.data.DataLoader(
        #     self.val_dataset,
        #     sampler=sampler_val,
        #     batch_size=int(1.5 * args.batch_size),
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=False,
        # )

        return data_loader_train, None

    def _create_model(self, args):
        args = self.args

        # Create model
        mae = mae_factory(
            args.model, args.baseline_model, args.window_size, args.stgm_location, True
        )

        if args.checkpoint:
            mae.load_state_dict(torch.load(args.checkpoint), strict=False)

        return mae

    def _create_optimizer(self, args):
        args = self.args
        # Create optimizer
        parameters = []
        if args.whole:
            parameters = self.model.training_parameters()

        parameters = torch.nn.ParameterList(parameters)
        return create_optimizer(args, parameters)
    

    def train(self):
        args = self.args
        logging.info(f"{args}")
        for epoch in tqdm(range(args.epochs)):
            # Train
            losses = self.train_epoch(args)
            # Scheduler Step
            self.lr_scheduler.step(epoch)
            # Log loss
            logging.info(
                f"Epoch {epoch}: {' '.join([f'{key}: {val}' for key, val in losses.items()])}"
            )
            # Create checkpoint
            checkpoint_path = args.save_loc / (
                f"epoch{epoch}" + f"ws{args.window_size}bl{args.bottleneck_location}.pt"
            )
            torch.save(self.model.state_dict(), checkpoint_path)

    def train_epoch(self, args):
        loss_mean = MeanMetric().to(device=self.fabric.device)
        self.model.train()

        for samples, targets in tqdm(self.data_loader_train):
            loss = self.model(samples)
            self.fabric.backward(loss)
            self.optimizer.step()
            loss_mean(loss)

        losses = {
            "batch_loss": loss_mean.compute(),
        }
        return losses


def main():
    CHECKPOINT_PATH = Path("./checkpoints")

    parser = get_args_parser()
    args = parser.parse_args()
    # Setup Logging
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINT_PATH / f"mae-{now}"
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
    train_eval.train()

if __name__ == "__main__":
    main()
