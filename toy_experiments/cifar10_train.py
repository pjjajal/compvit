import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from timm.scheduler import create_scheduler_v2
# import wandb

from spvit.spvit_model import SPViT

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


def eval(testloader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to("mps")
            labels = labels.to("mps")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="compvit-toy",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 1e-3,
    #         "architecture": "ViT",
    #         "dataset": "CIFAR-10",
    #         "epochs": 100,
    #     },
    # )
    dataset = CIFAR10("./toy_experiments", transform=TRANSFORM)
    eval_dataset = CIFAR10("./toy_experiments", transform=TRANSFORM_TEST, train=False)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=1)

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=6,
    ).to(device="mps")
    model = torch.compile(model, backend="onnxrt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    # scheduler, num_epochs = create_scheduler_v2(
    #     optimizer,
    #     sched="cosine",
    #     num_epochs=102,
    #     warmup_epochs=2,
    #     warmup_lr=1e-4,
    #     min_lr=8e-4,
    #     cycle_limit=10,
    # )

    for epoch in range(100):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device="mps")
            label = label.to(device="mps")

            optimizer.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        batch_loss = running_loss / len(train_loader)
        val_acc = eval(eval_loader, model)
        print(f"batch loss: {batch_loss}, val acc: {val_acc}")
        # wandb.log({"val_acc": val_acc, "train loss": batch_loss})
        torch.save(model.state_dict(), f"./toy_experiments/{epoch}_cifar10.pt")
    # wandb.finish()
