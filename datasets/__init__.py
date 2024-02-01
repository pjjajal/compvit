from .imagenet import create_imagenet_dataset
from .cifar import create_cifar100_dataset, create_cifar10_dataset


def create_dataset(args):
    train_dataset = None
    test_dataset = None
    if args.dataset == "imagenet":
        train_dataset = create_imagenet_dataset(
            split="train", data_dir=args.data_dir, pretraining=args.pretraining
        )
        test_dataset = (
            create_imagenet_dataset(
                split="val", data_dir=args.data_dir, pretraining=args.pretraining
            )
            if not args.pretraining
            else None
        )
    elif args.dataset == "cifar100":
        train_dataset, test_dataset = create_cifar100_dataset(args)
    elif args.dataset == "cifar10":
        train_dataset, test_dataset = create_cifar10_dataset(args)
    return train_dataset, test_dataset
