import torchvision.transforms as tvt

from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

TRANSFORM = tvt.Compose(
    [
        tvt.RandomCrop(32, padding=4),
        tvt.Resize(224),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

TRANSFORM_TEST = tvt.Compose(
    [
        tvt.Resize(224),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def create_cifar100_dataset(args):
    train_dataset = CIFAR100(args.data_dir, transform=TRANSFORM, download=True)
    test_dataset = CIFAR100(
        args.data_dir, transform=TRANSFORM_TEST, train=False, download=True
    )

    return train_dataset, test_dataset


def create_cifar10_dataset(args):
    train_dataset = CIFAR10(args.data_dir, transform=TRANSFORM, download=True)
    test_dataset = CIFAR10(
        args.data_dir, transform=TRANSFORM_TEST, train=False, download=True
    )

    return train_dataset, test_dataset
