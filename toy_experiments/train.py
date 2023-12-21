import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
import wandb
from omegaconf import OmegaConf
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from timm.data import Mixup
from tqdm import tqdm

from spvit.spvit_model import SPViT

TOY_EXPERIMENTS_PATH = Path("./toy_experiments")
DATA_PATH = TOY_EXPERIMENTS_PATH / "data"
CONFIG_PATH = TOY_EXPERIMENTS_PATH / "configs"
CHECKPOINTS_PATH = TOY_EXPERIMENTS_PATH / "checkpoints"


TRANSFORM = tvt.Compose(
    [
        tvt.RandomCrop(32, padding=4),
        tvt.Resize(32),
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
    parser.add_argument("--model", required=True, choices=["vit", "compvit"])
    parser.add_argument("--eval", action="store_true")

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


def evaluate(testloader, model):
    correct = 0
    total = 0
    elapsed_time = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs = model(images)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time += start_event.elapsed_time(end_event)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, elapsed_time / len(testloader)


def main(args):
    config_path = CONFIG_PATH / (args.dataset + ".yaml")
    configs = OmegaConf.load(config_path)
    model_config = configs[args.model]
    hyperparameters = configs["hyperparameters"]

    wandb.init(
        # set the wandb project where this run will be logged
        project="compvit-toy",
        # track hyperparameters and run metadata
        config={
            **model_config,
            **hyperparameters,
            "architecture": args.model,
            "dataset": args.dataset,
        },
    )

    # Dataset and Dataloaders
    train_dataset, test_dataset = create_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Mixup
    mixup_fn = Mixup(
        mixup_alpha=hyperparameters["mixup_alpha"],
        num_classes=model_config["num_classes"],
    )

    if args.model == "vit":
        model = VisionTransformer(**model_config)

    if args.model == "compvit":
        model = SPViT(**model_config)
        model.load_state_dict(
            torch.load(
                "toy_experiments/checkpoints/mae_2023-12-20-164956/best_performing.pt"
            )
        )

    model = model.to(device="cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=hyperparameters["lr"], weight_decay=0.05
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

    highest_val_acc = 0.0
    for epoch in range(hyperparameters["epochs"]):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device="cuda")
            label = label.to(device="cuda")
            img, label = mixup_fn(img, label)

            optimizer.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        batch_loss = running_loss / len(train_loader)
        val_acc, inf_time = evaluate(test_loader, model)
        
        # Logging
        print(f"batch loss: {batch_loss}, val acc: {val_acc}, inf time: {inf_time}, lr: {optimizer.param_groups[0]['lr']}")
        wandb.log({"val_acc": val_acc, "train loss": batch_loss, "inf time": inf_time, "lr": optimizer.param_groups[0]['lr']})
        
        # Scheduler Step
        if epoch >= hyperparameters['warmup_epochs']:
            cosine_scheduler.step()
        else:
            warmup_scheduler.step()
        
        # Save Model
        if val_acc > highest_val_acc:
            highest_val_acc = val_acc
            save_path = args.save_loc / f"best_performing.pt"
            torch.save(model.state_dict(), save_path)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    args.save_loc = save_loc
    main(args)
