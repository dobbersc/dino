from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn

Transform: TypeAlias = Callable[[Tensor], Tensor]


class Augmenter(nn.Module):
    """Utility class that applies a series of augmentation transforms to an input image."""

    def __init__(self, transforms: Transform | Sequence[Transform], num_augmentations: int | Sequence[int] = 1) -> None:
        """Initializes an Augmenter.

        :param transforms:
        :param num_augmentations:
        """
        super().__init__()

        self.transforms = transforms if isinstance(transforms, Sequence) else (transforms,)
        self.num_augmentations = (
            num_augmentations
            if isinstance(num_augmentations, Sequence)
            else (num_augmentations,) * len(self.transforms)
        )

        if len(self.transforms) != len(self.num_augmentations):
            msg: str = (
                f"The number of transforms ({len(self.transforms)}) must match the number of "
                f"augmentations to apply ({len(self.num_augmentations)})."
            )
            raise ValueError(msg)

        if any(n < 1 for n in self.num_augmentations):
            msg = ""  # TODO: Add error message
            raise ValueError(msg)

    def forward(self, image: Tensor) -> Tensor:
        """

        :param image:
        :return:
        """
        return torch.stack(
            [
                transform(image)
                for transform, n in zip(self.transforms, self.num_augmentations, strict=True)
                for _ in range(n)
            ],
            dim=0,
        )


class DefaultLocalAugmenter(Augmenter):
    def __init__(self, num_augmentations: int = 8) -> None:
        # TODO: Integrate default local transformations
        super().__init__(transforms=lambda x: x, num_augmentations=num_augmentations)


class DefaultGlobalAugmenter(Augmenter):
    def __init__(self) -> None:
        # TODO: Integrate default global transformations
        first_global_transform = lambda x: x  # noqa: E731
        second_global_transform = lambda x: x  # noqa: E731
        super().__init__(transforms=[first_global_transform, second_global_transform], num_augmentations=(1, 1))
