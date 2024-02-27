from typing import List, Tuple

import numpy as np
import torch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    ImageMixup,
    LabelMixup,
    NormalizeImage,
    RandomHorizontalFlip,
    Squeeze,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .augments import (
    GaussianBlur,
    RandomColorJitter,
    RandomGrayscale,
    RandomSolarization,
)

# Define ImageNet and Std for noramlization.
IMAGENET_MEAN = np.array(IMAGENET_DEFAULT_MEAN) * 255
IMAGENET_STD = np.array(IMAGENET_DEFAULT_STD) * 255

# Define default crop ratio (taken from ffcv https://github.com/libffcv/ffcv-imagenet/blob/4dd291a122096a2032ec9fd19633e1b9aff3577d/train_imagenet.py#L94).
DEFAULT_CROP_RATIO = 224 / 256


# Define pipelines
def create_train_pipeline(
    device: str = "cuda",
    pretraining: bool = True,
    input_size: int = 224,
    **kwargs,
) -> Tuple[List[Operation], List[Operation]]:
    ########################################################
    # Image pipeline
    decoder = RandomResizedCropRGBImageDecoder((input_size, input_size))
    if pretraining:
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToTorchImage(convert_back_int16=False),
            ToDevice(device, non_blocking=True),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]
    else:
        # Image pipeline with 3Aug
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            RandomGrayscale(p=1.0),
            GaussianBlur(p=1.0),
            RandomSolarization(p=1.0),
            RandomColorJitter(p=0.3),  # Hardcoded value from DEIT3 3aug
        ]

        # Mixup
        if "mixup_alpha" in kwargs:
            image_pipeline.append(ImageMixup(alpha=kwargs["mixup_alpha"]))

        # Add the rest of the pipeline
        image_pipeline.extend(
            [
                ToDevice(device, non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]
        )
    ########################################################
    ########################################################
    # Label pipeline
    label_pipeline = [IntDecoder()]
    # Label Mixup
    if "mixup_alpha" in kwargs:
        label_pipeline.append(LabelMixup(alpha=kwargs["mixup_alpha"]))
    # Rest of label pipeline
    label_pipeline.extend(
        [
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ]
    )
    ########################################################

    return image_pipeline, label_pipeline


def create_val_pipeline(
    device: str = "cuda",
    input_size: int = 224,
):
    # Image pipeline
    decoder = CenterCropRGBImageDecoder(
        (input_size, input_size), ratio=DEFAULT_CROP_RATIO
    )
    image_pipeline: List[Operation] = [
        decoder,
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    # Label pipeline
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True),
    ]
    return image_pipeline, label_pipeline
