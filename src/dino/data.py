from collections.abc import Sized
from typing import NamedTuple

from torch import Tensor
from torch.utils.data import Dataset

from dino.augmentation import Augmenter


class Augmentations(NamedTuple):
    local_augmentations: Tensor
    global_augmentations: Tensor


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
