import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset


# Function to separate data by class
def separate_by_class(dataset):
    data_by_class = defaultdict(list)
    for data, target in dataset:
        data_by_class[target].append((data, target))
    return data_by_class


def select_percentage(data_by_class, percentage):
    selected_data = []
    for class_idx, data in data_by_class.items():
        class_size = len(data)
        select_size = int(class_size * percentage)
        selected_data.extend(data[:select_size])
    return selected_data


class PMNISTDataset(Dataset):
    def __init__(self, data, k):
        self.data = data
        self.k = k  # Number of random augmentations
        self.augmented_data = self.augment_data()

    def augment(self, image):
        augmented_images = []
        for i in range(self.k):
            shift_x, shift_y = random.randint(0, 14), random.randint(0, 14)
            aug_img = torch.roll(image, shifts=(shift_x, shift_y), dims=(1, 2))
            augmented_images.append(aug_img)

        return augmented_images

    def augment_data(self):
        augmented_data = []
        for image, label in self.data:
            augmented_images = self.augment(image)
            augmented_data.extend([(aug_img, label) for aug_img in augmented_images])
        return augmented_data

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        return self.augmented_data[idx]
