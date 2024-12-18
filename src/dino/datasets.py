import logging
import os
from collections import defaultdict
from collections.abc import Callable, Generator, Sized
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import PIL
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms  # type: ignore
from torchvision.datasets import CIFAR10, ImageFolder

from dino.augmentation import Augmenter

logger = logging.getLogger(__name__)

_T_co = TypeVar("_T_co", covariant=True)


class DatasetType(Enum):
    CUSTOM_IMAGENET = "CUSTOM_IMAGENET"
    IMAGE_FOLDER = "IMAGE_FOLDER"
    CIFAR10 = "CIFAR10"


class TransformType(Enum):
    DEFAULT = "default"
    LINEAR_VAL = "linear_val"
    LINEAR_TRAIN = "linear_train"
    KNN = "knn"


@dataclass
class DatasetConfig:
    type_: DatasetType = DatasetType.IMAGE_FOLDER
    transform: TransformType = TransformType.DEFAULT
    data_dir: str = Path.cwd() / "data"
    train: bool = True


@dataclass
class CustomImageNetConfig(DatasetConfig):
    sample_classes_indices: tuple[int, ...] | None = None
    path_wnids: str | None = None


_cs = ConfigStore.instance()
_cs.store(
    group="dataset",
    name="base_dataset",
    node=DatasetConfig,
)
_cs.store(
    group="dataset",
    name="base_custom_imagenet",
    node=CustomImageNetConfig,
)


regular_transform: Callable[[Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224 for ImageNet models
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),  # ImageNet normalization
    ],
)
"""The default transform for ImageNet images."""

val_transform: Callable[[Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ],
)
"""The transform for linear evaluation on ImageNet validation set as described in the DINO paper."""

linear_train_transform: Callable[[Image], torch.Tensor] = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ],
)
"""The transform for linear evaluation on ImageNet training set as described in the DINO paper."""


def get_transform(transform_type: TransformType) -> Callable[[Image], torch.Tensor]:
    match transform_type:
        case TransformType.DEFAULT:
            return regular_transform
        case TransformType.LINEAR_VAL:
            return val_transform
        case TransformType.KNN:
            return val_transform
        case TransformType.LINEAR_TRAIN:
            return linear_train_transform


def get_dataset(cfg: DatasetConfig) -> Dataset[tuple[Image | torch.Tensor, int]]:
    """Returns the dataset based on the provided configuration."""
    match cfg.type_:
        case DatasetType.IMAGE_FOLDER:
            dataset = ImageFolder(
                root=cfg.data_dir,
                transform=get_transform(cfg.transform),
            )
            return split_imagefolder(dataset, train=cfg.train)
        case DatasetType.CUSTOM_IMAGENET:
            return ImageNetDirectoryDataset(
                data_dir=cfg.data_dir,
                transform=get_transform(cfg.transform),
                # path_wnids=cfg.path_wnids,
                # sample_classes_indices=cfg.sample_classes_indices,
                train=cfg.train,
            )
        case DatasetType.CIFAR10:
            return CIFAR10(
                # just specify the data directory that contains
                # 'cifar-10-batches-py' without the latter part
                root=cfg.data_dir,
                train=cfg.train,
                transform=get_transform(cfg.transform),
                download=False,
            )
            # maybe also get test dataset here (train=False) and concatenate them
        case _:
            msg = f"Invalid dataset type: {cfg.type_}"
            raise ValueError(msg)


def split_imagefolder(
    dataset: ImageFolder,
    split_ratio: float = 0.9,
    train: bool = True,
):
    """Splits deterministically into train/val subsets while maintaining balanced class labels."""
    # Group indices by class
    train_indices, val_indices = get_split_indices(dataset.samples, split_ratio)

    # Return the appropriate subset
    if train:
        return ImageFolderSubset(dataset, train_indices)
    return ImageFolderSubset(dataset, val_indices)


class ImageFolderSubset(Subset):
    """A Subset of ImageFolder that retains the `classes` property."""

    def __init__(self, dataset: ImageFolder, indices: list[int]):
        super().__init__(dataset, indices)
        self.classes = dataset.classes  # Retain the classes property
        self.class_to_idx = dataset.class_to_idx  # Retain the class_to_idx mapping

    @property
    def samples(self):
        """Return the samples in this subset."""
        return [self.dataset.samples[i] for i in self.indices]


def get_split_indices(
    samples: list[tuple[Path, int]],
    split_ratio: float,
) -> tuple[list[int], list[int]]:
    """Splits the samples into train/val indices while maintaining balanced class labels."""
    class_to_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(samples):
        class_to_indices[class_idx].append(idx)

    for class_idx in class_to_indices:
        class_to_indices[class_idx].sort(key=lambda idx: samples[idx][0])

    train_indices = []
    val_indices = []

    for indices in class_to_indices.values():
        split_point = int(len(indices) * split_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    return train_indices, val_indices


# Might still be useful for visualization purposes or other interactions
@DeprecationWarning
class ImageNetDirectoryDataset(Dataset[tuple[Image | torch.Tensor, int]]):
    """A PyTorch Dataset for ImageNet images stored in directories."""

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable[[Image], torch.Tensor] | None = None,
        path_wnids: str | Path | None = None,
        sample_classes_indices: tuple[int, ...] | None = None,
        train: bool = True,  # New flag for train/validation split
        split_ratio: float = 0.9,  # Ratio for train split
    ):
        """Initializes an ImageNetDirectoryDataset.

        Args:
            data_dir: The directory containing the ImageNet images.
            transform: The transform to apply to the images. Default is None.
            path_wnids: The path to the file containing the class to words mapping. Default is None.
            num_sample_classes: The number of classes to sample from the dataset. Default is None.
            train: If True, use the training split, otherwise use validation split. Default is True.
            split_ratio: The ratio of the dataset to use for training. Default is 0.9.
        """
        self.data_dir = data_dir
        self.transform = transform
        samples_raw: list[tuple[Path, str]] = list(self.load_raw_samples(data_dir))

        # first then samples from raw
        samples_raw = sorted(
            samples_raw,
            key=lambda x: x[0],  # sort by path
        )

        first_samples = samples_raw[:5]
        msg = f"First samples: {first_samples}"
        logger.info(msg)

        self.wnid_to_class_idx = self.get_wnid_to_class_mapping(samples_raw)
        if sample_classes_indices is not None:
            samples_raw = list(
                filter(
                    lambda s: self.wnid_to_class_idx[s[1]] in sample_classes_indices,
                    samples_raw,
                ),
            )
            self.wnid_to_class_idx = self.get_wnid_to_class_mapping(samples_raw)

        self.class_idx_to_wnid = {v: k for k, v in self.wnid_to_class_idx.items()}

        all_samples = [(s[0], self.wnid_to_class_idx[s[1]]) for s in samples_raw]

        train_indices, val_indices = get_split_indices(all_samples, split_ratio)

        self.samples = (
            [all_samples[i] for i in train_indices]
            if train
            else [all_samples[i] for i in val_indices]
        )

        # for debugging
        if sample_classes_indices is not None:
            sample_classes = {s[1] for s in self.samples}
            class_names = {c_idx: self.class_idx_to_wnid[c_idx] for c_idx in sample_classes}
            msg = (
                f"Sampled {len(self.samples)} samples from {sample_classes_indices} "
                f"resulting classes: {class_names}"
            )
            logger.info(msg)

        if path_wnids is not None:
            self.class_idx_to_label = self.load_class_to_words(path_wnids, self.wnid_to_class_idx)

    @staticmethod
    def load_raw_samples(data_dir: str | Path) -> Generator[tuple[Path, str]]:
        """Loads the raw samples from the ImageNet directory structure."""
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
                        yield (file_path, wnid_name)

    @staticmethod
    def get_wnid_to_class_mapping(raw_samples: list[tuple[Path, str]]) -> dict[str, int]:
        """Creates a mapping from wnid to class index."""
        wnids_sorted = sorted({wnid for _, wnid in raw_samples})
        return {wnid_name: class_idx for class_idx, wnid_name in enumerate(wnids_sorted)}

    @staticmethod
    def load_class_to_words(path: str | Path, wnid_to_class_idx: dict[str, int]) -> dict[int, str]:
        """Loads the class to words mapping from the provided file."""
        class_idx_to_label: dict[int, str] = {}
        with open(path) as f:
            for line in f:
                wnid, label = line.strip().split("\t", 1)  # Split each line by tab
                if wnid in wnid_to_class_idx:
                    class_idx = wnid_to_class_idx[wnid]
                    class_idx_to_label[class_idx] = label
        return class_idx_to_label

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image | torch.Tensor, int]:
        """Returns the image and label at the specified index."""
        image_path, label = self.samples[idx]
        # Open image
        image = PIL.Image.open(image_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)  # type: ignore[assignment]

        return image, label

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        return len(self.wnid_to_class_idx)

    def get_class_name(self, class_idx: int) -> str | None:
        """Returns the class name for the specified class index.

        Args:
            class_idx: The class index to retrieve the name
        """
        return self.class_idx_to_label.get(class_idx, None)

    def get_image_by_class(self, class_idx: int) -> Generator[Image | torch.Tensor]:
        """Returns a generator of images for the specified class index.

        Args:
            class_idx: The class index to filter the images.
        """
        for idx, s in enumerate(self.samples):
            if s[1] == class_idx:
                yield self.__getitem__(idx)[0]


def unnorm(img: torch.Tensor) -> torch.Tensor:
    """Unnormalizes the image tensor for display."""
    # Unnormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std[:, None, None] + mean[:, None, None]  # Unnormalize
    return img.clamp(0, 1)


class UnlabelledDataset(Dataset[_T_co], Sized):
    """Wraps PyTorch Datasets of labelled data, removing the labels."""

    def __init__(self, dataset: Dataset[tuple[_T_co, Any]]) -> None:
        """Initializes an UnlabelledDataset.

        Args:
            dataset: The dataset containing the labelled data as tuples of data and their label.
        """
        if not isinstance(dataset, Sized):
            msg: str = "The provided dataset must implement the Sized interface, i.e. the '__len__' method."
            raise TypeError(msg)

        self.dataset = dataset

    def __getitem__(self, index: int) -> _T_co:
        return self.dataset[index][0]

    def __len__(self) -> int:
        return len(self.dataset)


class Views(NamedTuple):
    """Represents a list of local and global views of an image."""

    local_views: list[Tensor]
    global_views: list[Tensor]


class ViewDataset(Dataset[Views], Sized):
    """Wraps PyTorch Datasets, generating local and global views for each image in the original dataset."""

    def __init__(
        self,
        dataset: Dataset[Tensor],
        local_augmenter: Augmenter,
        global_augmenter: Augmenter,
    ) -> None:
        """Initializes a ViewDataset.

        Args:
            dataset: The dataset containing the original images.
            local_augmenter: The augmentation strategy to generate the local views.
            global_augmenter: The augmentation strategy to generate the global views.
        """
        if not isinstance(dataset, Sized):
            msg: str = "The provided dataset must implement the Sized interface, i.e. the '__len__' method."
            raise TypeError(msg)

        self.dataset = dataset
        self.local_augmenter = local_augmenter
        self.global_augmenter = global_augmenter

    def __getitem__(self, index: int) -> Views:
        image: Tensor = self.dataset[index]
        return Views(
            local_views=self.local_augmenter(image),
            global_views=self.global_augmenter(image),
        )

    def __len__(self) -> int:
        return len(self.dataset)
