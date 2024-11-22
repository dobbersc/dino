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

# %% [markdown]
# # Evaluate dataset
#
# 1. load eval dataset
# 2. load train dataset
# 3. load model
# 4. run feature extraction on both datasets
# 5. knn_acc

# %%
import random

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dino.datasets import ImageNetDirectoryDataset, regular_transform

# %%
DATA_DIR = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/train"
# transform = T.ToImage()
transform = regular_transform
N_SAMPLES = 2

train_ds = ImageNetDirectoryDataset(
    data_dir=DATA_DIR,
    transform=transform,
    num_sample_classes=N_SAMPLES,
    train=True,
)

small_subset_size = 50
small_subset_indices = random.sample(
    list(range(len(train_ds))),
    small_subset_size,
)

# Create the smaller subset
train_ds = Subset(train_ds, small_subset_indices)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_ds = ImageNetDirectoryDataset(
    data_dir=DATA_DIR,
    transform=transform,
    num_sample_classes=N_SAMPLES,
    train=False,
)
small_subset_size = 10
small_subset_indices = random.sample(
    list(range(len(train_ds))),
    small_subset_size,
)

# Create the smaller subset
val_ds = Subset(val_ds, small_subset_indices)
val_loader = DataLoader(val_ds, batch_size=62, shuffle=True)

# %%
print(f"Train dataset: {len(train_ds)} samples")
print(f"Val dataset: {len(val_ds)} samples")

# %%
model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")


# %%
def extract_features(
    model,
    loader,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features, targets = ([], [])
    for images, targets_ in tqdm(loader):
        output = model(images)
        features.extend(output)
        targets.extend(targets_)
    return features, targets



# %%
train_features, train_targets = extract_features(model, train_loader)
