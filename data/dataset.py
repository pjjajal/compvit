from pathlib import Path
from typing import Sequence
import os

import numpy as np
import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, ImageFolder
from .augment import new_data_aug_generator

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = tvt.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            tvt.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(tvt.CenterCrop(args.input_size))

    t.append(tvt.ToTensor())
    t.append(tvt.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return tvt.Compose(t)


def create_imagenet_dataset(split: str, cache_dir: Path, args, mae=False):
    if split == "train":
        # _transform = make_classification_train_transform()
        if mae:
            _transform = tvt.Compose(
                [
                    tvt.RandomResizedCrop(
                        args.input_size, scale=(0.2, 1.0), interpolation=3
                    ),  # 3 is bicubic
                    tvt.RandomHorizontalFlip(),
                    tvt.ToTensor(),
                    tvt.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ]
            )
        else:
            _transform = build_transform(True, args)
    elif split == "val":
        _transform = build_transform(False, args)

    ### MINOR CHANGE
    ### Do not use torchvision ImageNet - Instead use unzipped ImageNet1K with vanilla ImageFolder loading
    #dataset = ImageNet(cache_dir, split, transform=_transform)
    dataset  = ImageFolder(root=os.path.join(cache_dir, split), transform=_transform)

    return dataset

if __name__ == "__main__":
    dataset = create_imagenet_dataset("val", cache_dir="E:\datasets\imagenet")
    dataloader = DataLoader(dataset, 32, False, pin_memory=True)
    print(next(iter(dataloader)))
