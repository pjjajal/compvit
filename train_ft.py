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
from timm.data import Mixup


import wandb
from datasets import create_dataset
from dinov2.factory import dinov2_factory
from compvit.factory import compvit_factory


torch.set_float32_matmul_precision("medium")

CONFIG_PATH = Path("./configs")
CHECKPOINTS_PATH = Path("./checkpoints")


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


def evaluate(test_loader, model, head):
    correct = 0
    total = 0
    elapsed_time = 0.0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs = model(images, is_training=True)
            if args.model == "dinov2":
                patch_tokens = outputs["x_norm_patchtokens"]
                cls_token = outputs["x_norm_clstoken"]
                outputs = torch.cat([
                    cls_token,
                    patch_tokens.mean(dim=1),
                ], dim=1)
            elif args.model == "compvit":
                outputs = outputs["x_norm"].mean(-2)
            outputs = head(outputs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time += start_event.elapsed_time(end_event)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, elapsed_time / len(test_loader)

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
            **hyperparameters,
            **args,
        },
    )

    # Create dataset and train loader.
    train_dataset, test_dataset = create_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=hyperparameters["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparameters["test_batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Setup dataloader on Fabric.
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Mixup
    mixup_fn = Mixup(
        mixup_alpha=hyperparameters["mixup_alpha"],
        num_classes=head_config["num_classes"],
    )

    if args.model == "dinov2":
        model, config = dinov2_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading weights")
            model.load_state_dict(torch.load(model_config["checkpoint"]))

    if args.model == "compvit":
        model, config = compvit_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading", model_config["checkpoint"])
            model.load_state_dict(torch.load(model_config["checkpoint"]))

    # Update W&B run metadata.
    run.config.update({**config})

    head = nn.Linear(model.embed_dim * 2, head_config["num_classes"])
    if head_config["checkpoint"]:
        print("Loading", head_config["checkpoint"])
        head.load_state_dict(torch.load(head_config["checkpoint"]))

    # model = fabric.setup_module(model)
    # head = fabric.setup_module(head)

    # Create optimizer and scheduler.
    criterion = nn.CrossEntropyLoss()
    parameters = []
    parameters.extend(head.parameters())
    if args.model == "compvit" and not args.head:
        parameters.extend(model.parameters())
    optimizer = optim.AdamW(
        parameters,
        lr=hyperparameters["lr"],
        weight_decay=0.05,
    )
    if args.model == "compvit" and not args.head:
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            hyperparameters["epochs"] - hyperparameters["warmup_epochs"],
            hyperparameters["min_lr"],
            verbose=True,
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=hyperparameters["warmup_lr_scale"],
            end_factor=1.0,
            total_iters=hyperparameters["warmup_epochs"],
            verbose=True,
        )

    # Setup model and optimizer.
    model, optimizer = fabric.setup(model, optimizer)
    head = fabric.setup_module(head)
    # optimizer = fabric.setup_optimizers(optimizer)


    # EVAL STUFF ########
    if args.eval:
        print("Evaluating")
        val_acc, inf_time = evaluate(test_loader, model, head)
        print(f"val acc: {val_acc}, inf time: {inf_time}")
        wandb.log({"val_acc": val_acc, "inf time": inf_time, "eval": True})
        wandb.finish()
        return 0
    ####################

    highest_val_acc = 0.0
    for epoch in range(hyperparameters["epochs"]):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            is_accumulating = (i + 1) % hyperparameters["accumulations"] != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # Mixup
                img, label = mixup_fn(img, label)
                # Forward pass.
                outputs = model(img, is_training=True)
                if args.model == "dinov2":
                    patch_tokens = outputs["x_norm_patchtokens"]
                    cls_token = outputs["x_norm_clstoken"]
                    outputs = torch.cat([
                        cls_token,
                        patch_tokens.mean(dim=1),
                    ], dim=1)
                elif args.model == "compvit":
                    outputs = outputs["x_norm"].mean(-2)
                outputs = head(outputs)
                loss = criterion(outputs, label)
                # Running loss.
                running_loss += loss.detach().item()
                # Backward pass.
                fabric.backward(loss)
            if not is_accumulating or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        batch_loss = running_loss / len(train_loader)
        val_acc, inf_time = evaluate(test_loader, model, head)

        # Logging
        print(
            f"batch loss: {batch_loss}, val acc: {val_acc}, inf time: {inf_time}, lr: {optimizer.param_groups[0]['lr']}"
        )
        wandb.log(
            {
                "val_acc": val_acc,
                "train loss": batch_loss,
                "inf time": inf_time,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Scheduler Step
        if args.model == "compvit" and not args.head:
            if epoch >= hyperparameters["warmup_epochs"]:
                cosine_scheduler.step()
            else:
                warmup_scheduler.step()

        # Save Model
        if val_acc > highest_val_acc:
            highest_val_acc = val_acc
            save_path = args.save_loc / f"best_performing.pt"
            head_save_path = args.save_loc / f"best_performing_head.pt"
            torch.save(model.state_dict(), save_path)
            torch.save(head.state_dict(), head_save_path)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = CHECKPOINTS_PATH / now
    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = False
    main(args)
