import os

import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from .augmentation_torus import PMNISTDataset, select_percentage, separate_by_class
from .augmentation_utils import Augmentation, CutoutDefault, autoaug_paper_cifar10
from .dataset import LinearDataset


def get_dataset(dataset_name, config):
    if dataset_name == "LinearDataset":
        dataset = LinearDataset(
            input_dim=config.data.input_dim,
            output_dim=config.data.output_dim,
            noise_std=config.data.noise_std,
            data_seed=config.data.data_seed,
            sample_seed=config.data.sample_seed,
            num_samples=config.data.train_samples,
            distribution=config.data.distribution,
            normalize=config.data.normalize,
        )
        train_dataset = dataset
        # Using internal method to generate validation set
        val_X, val_y = dataset._generate_samples(config.data.val_samples)
        validation_dataset = TensorDataset(val_X, val_y)
    elif dataset_name == "MNIST":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_dataset = MNIST(
            os.environ["DATA_DIR"], train=True, transform=transform, download=True
        )
        validation_dataset = MNIST(
            os.environ["DATA_DIR"], train=False, transform=transform
        )
    elif dataset_name == "MNIST_14_Torus":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose(
            [transforms.Resize((14, 14)), transforms.ToTensor(), normalize]
        )

        train_dataset = MNIST(
            os.environ["DATA_DIR"], train=True, transform=transform, download=True
        )
        validation_dataset = MNIST(
            os.environ["DATA_DIR"], train=False, transform=transform
        )

        data_by_class_train = separate_by_class(train_dataset)
        data_by_class_test = separate_by_class(validation_dataset)

        selected_train_data = select_percentage(
            data_by_class_train, percentage=config.data.p
        )
        selected_test_data = select_percentage(data_by_class_test, percentage=1)

        train_dataset = PMNISTDataset(selected_train_data, k=config.data.k)
        validation_dataset = PMNISTDataset(selected_test_data, k=1)

    elif dataset_name == "CIFAR10":
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ]
        )

        transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        transform_train.transforms.append(CutoutDefault(16))

        train_dataset = CIFAR10(
            os.environ["DATA_DIR"], train=True, transform=transform_train, download=True
        )
        validation_dataset = CIFAR10(
            os.environ["DATA_DIR"], train=False, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    return train_dataset, validation_dataset
