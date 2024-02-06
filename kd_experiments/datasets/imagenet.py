from pathlib import Path
from typing import Sequence

import numpy as np
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from omegaconf import OmegaConf
from .augment import new_data_aug_generator
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


def make_classification_train_transform(
    *,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Compose:
    transforms_list = [
        tvt.RandomResizedCrop(224),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return tvt.Compose(transforms_list)


def make_3aug_train_transform() -> tvt.Compose:
    args = OmegaConf.create(
        {
            "input_size": 224,
            "src": False,
            "color_jitter": 0.3,  # This is set in the CLI of DEIT
        }
    )
    return new_data_aug_generator(args)


def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=tvt.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Compose:
    transforms_list = [
        tvt.Resize(resize_size, interpolation=interpolation),
        tvt.CenterCrop(crop_size),
        tvt.ToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return tvt.Compose(transforms_list)


def create_imagenet_dataset(split: str, data_dir: Path, pretraining: bool = True, eval_transform=None):
    if split == "train":
        _transform = (
            make_classification_train_transform()
            if pretraining
            else make_3aug_train_transform()
        )
    elif split == "val":
        _transform = make_classification_eval_transform()
        if eval_transform:
            _transform = eval_transform

    dataset = ImageNet(data_dir, split, transform=_transform)
    return dataset


if __name__ == "__main__":
    dataset = create_imagenet_dataset("val", cache_dir="E:\datasets\imagenet")
    dataloader = DataLoader(dataset, 32, False, pin_memory=True)
    print(next(iter(dataloader)))
