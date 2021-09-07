from typing import Tuple

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def load_data() -> Tuple[CIFAR10, CIFAR10]:
    mean, std = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
    norm_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    training_data = CIFAR10(
        root="data", 
        train=True, 
        download=True,
        transform=transforms.Compose(norm_transforms))
    validation_data = CIFAR10(
        root="data", 
        train=False,
        download=True,
        transform=transforms.Compose(norm_transforms))
    return (training_data, validation_data)
