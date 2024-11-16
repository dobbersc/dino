# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

# %%
# Define the transform (preprocessing) for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),               # Convert PIL image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the images
])

# Load the CIFAR-10 training and test datasets
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",         # Directory where the dataset will be stored
    train=True,            # Load training set
    download=True,         # Download the dataset if it is not already downloaded
    # transform=transform    # Apply transformations
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",         # Directory where the dataset will be stored
    train=False,           # Load test set
    download=True,         # Download the dataset if it is not already downloaded
    transform=transform,    # Apply transformations
)

# Create data loaders for batching and shuffling
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2,
)

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=64, shuffle=False, num_workers=2
# )


# %%
len(train_dataset)

# %%
transform = transforms.Compose([
    transforms.Resize(256),                  # Resize to 256x256 pixels
    transforms.CenterCrop(224),              # Crop to 224x224 (common for ImageNet models)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

imagenet_root = Path("/Users/pvmeng/Documents/dino/data/tiny-imagenet-200")
cifar_root = Path("/Users/pvmeng/Documents/dino/data/")
# Load the training dataset
cifar_train = CIFAR10(
    root=cifar_root,
    train=True,
    transform=transform,
    download=False,
)
cifar_val = CIFAR10(
    root=cifar_root,
    train=False,
    transform=transform,
    download=False,
)
# merge the training and validation sets
cifar_ds = torch.utils.data.ConcatDataset([cifar_train, cifar_val])

# %%
cifar_val.classes
# {train_dataset[i][1] for i in range(len(train_dataset))}
# train_dataset.classes[0]

# %%
