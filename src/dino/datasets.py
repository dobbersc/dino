import os
import random
from collections.abc import Callable, Sized, Sequence
from pathlib import Path
from typing import NamedTuple

import PIL
import torch
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from dino.augmentation import Augmenter


class ImageNetDirectoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable[[Image], torch.Tensor] | None = None,
        path_wnids: str | Path | None = None,
        num_sample_classes: int | None = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Iterate over directories (each representing a wnid) in the main data directory
        for wnid_dir in os.listdir(data_dir):
            wnid_path = Path(data_dir) / wnid_dir
            # Ensure it's a directory (ignore non-directory files)
            if wnid_path.is_dir():
                # get the wnid name, the last part of the path
                wnid_name = wnid_path.parts[-1]

                if wnid_name.endswith(".tar"):
                    wnid_name = wnid_name[:-4]

                # check if there is another directory inside the wnid directory called images
                if (wnid_path / "images").is_dir():
                    wnid_path = wnid_path / "images"
                # Collect each image file in the wnid directory
                for file_name in os.listdir(wnid_path):
                    if file_name.endswith(".JPEG"):  # Filter for image files
                        file_path = wnid_path / file_name
                        self.samples.append((file_path, wnid_name))

        # Create a mapping from wnid name to integer label
        self.wnid_to_class_idx = {
            wnid_name: class_idx for class_idx, wnid_name in enumerate({s[1] for s in self.samples})
        }

        if num_sample_classes is not None:
            # smaple class_indices for the subset
            class_indices = random.sample(range(len(self.wnid_to_class_idx)), num_sample_classes)
            # filter samples for the subset
            self.samples = [s for s in self.samples if self.wnid_to_class_idx[s[1]] in class_indices]
            # update the mapping
            self.wnid_to_class_idx = {
                wnid_name: class_idx for class_idx, wnid_name in enumerate({s[1] for s in self.samples})
            }

        self.class_idx_to_wnid = {v: k for k, v in self.wnid_to_class_idx.items()}
        self.samples = [(s[0], self.wnid_to_class_idx[s[1]]) for s in self.samples]

        if path_wnids is not None:
            self.class_idx_to_label = {}
            with open(path_wnids) as f:
                for line in f:
                    wnid, label = line.strip().split("\t", 1)  # Split each line by tab
                    if wnid in self.wnid_to_class_idx:
                        class_idx = self.wnid_to_class_idx[wnid]
                        self.class_idx_to_label[class_idx] = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image | torch.Tensor, int]:
        image_path, label = self.samples[idx]
        # Open image
        image = PIL.Image.open(image_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, class_idx: int) -> str | None:
        return self.class_idx_to_label.get(class_idx, None)

    def get_image_by_class(self, class_idx: int) -> Image | torch.Tensor:
        for idx, s in enumerate(self.samples):
            if s[1] == class_idx:
                yield self.__getitem__(idx)[0]


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224 for ImageNet models
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ]
)


def unnorm(img: torch.Tensor) -> torch.Tensor:
    # Unnormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std[:, None, None] + mean[:, None, None]  # Unnormalize
    return img.clamp(0, 1)


class Augmentations(NamedTuple):
    local_augmentations: list[Tensor]
    global_augmentations: list[Tensor]


class AugmentedDataset(Dataset[Augmentations], Sized):
    def __init__(self, dataset: Dataset[Tensor], local_augmenter: Augmenter, global_augmenter: Augmenter) -> None:
        if not isinstance(dataset, Sized):
            msg: str = "The provided dataset must implement the Sized interface, i.e. the '__len__' method."
            raise TypeError(msg)

        self.dataset = dataset
        self.local_augmenter = local_augmenter
        self.global_augmenter = global_augmenter

    def __getitem__(self, index: int) -> Augmentations:
        image: Tensor = self.dataset[index]
        return Augmentations(
            local_augmentations=self.local_augmenter(image),
            global_augmentations=self.global_augmenter(image),
        )

    def __len__(self) -> int:
        return len(self.dataset)
