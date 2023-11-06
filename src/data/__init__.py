from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

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
        transform = transforms.ToTensor()  # Add necessary transforms
        train_dataset = MNIST("./data", train=True, transform=transform, download=True)
        validation_dataset = MNIST("./data", train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    return train_dataset, validation_dataset
