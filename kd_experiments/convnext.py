import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm


def evaluate(testloader, model, head):
    correct = 0
    total = 0
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = model.features(img)
            outputs = model.avgpool(outputs)
            outputs = head(outputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model = model.train()
    return correct / total


if __name__ == "__main__":
    transform_train = tvt.Compose(
        [
            tvt.Resize((224, 224)),
            tvt.RandomHorizontalFlip(0.1),
            # tvt.RandomRotation(20),
            tvt.ToTensor(),
            # tvt.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
            # tvt.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # tvt.RandomErasing(p=0.75,scale=(0.02, 0.1),value = 1.0, inplace = False)
        ]
    )

    transform_test = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Resize((224, 224)),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="toy_experiments/data",
        train=True,
        download=True,
        transform=transform_train,
    )

    crossdataset = torchvision.datasets.CIFAR100(
        root="toy_experiments/data",
        train=False,
        download=True,
        transform=transform_test,
    )

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        crossdataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    device = "mps"
    model = torchvision.models.convnext_tiny(
        weights=torchvision.models.ConvNeXt_Tiny_Weights
    ).to(device=device)

    head = nn.Sequential(
        torchvision.models.convnext.LayerNorm2d((768,)),
        nn.Flatten(1),
        nn.Linear(768, 100),
    ).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.05)

    highest_val_acc = 0.0
    for epoch in range(10):
        running_loss = 0.0
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device=device)
            label = label.to(device=device)

            outputs = model.features(img)
            outputs = model.avgpool(outputs)
            outputs = head(outputs)
            loss = criterion(outputs, label)

            running_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        batch_loss = running_loss / len(train_loader)
        val_acc, inf_time = evaluate(test_loader, model, head)

        # Logging
        print(
            f"batch loss: {batch_loss}, val acc: {val_acc}"
        )
