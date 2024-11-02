from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn

Transform: TypeAlias = Callable[[Tensor], Tensor]


class Augmenter(nn.Module):
    """Utility class that applies a series of augmentation transforms to an input image."""

    def __init__(self, transforms: Transform | Sequence[Transform], repeats: int | Sequence[int] = 1) -> None:
        """Initializes an Augmenter.

        :param transforms:
        :param repeats:
        """
        super().__init__()

        self.transforms = transforms if isinstance(transforms, Sequence) else (transforms,)
        self.repeats = repeats if isinstance(repeats, Sequence) else (repeats,) * len(self.transforms)

        if len(self.transforms) != len(self.repeats):
            # TODO: Better error message
            msg: str = (
                f"Mismatch between number of transforms ({len(self.transforms)}) and "
                f"number of their repeats ({len(self.repeats)})."
            )
            raise ValueError(msg)

        if any(n < 1 for n in self.repeats):
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
                for transform, repeat in zip(self.transforms, self.repeats, strict=True)
                for _ in range(repeat)
            ],
            dim=0,
        )


class DefaultLocalAugmenter(Augmenter):
    def __init__(self, repeats: int = 8) -> None:
        # TODO: Integrate default local transformations
        super().__init__(transforms=lambda x: x, repeats=repeats)


class DefaultGlobalAugmenter(Augmenter):
    def __init__(self) -> None:
        # TODO: Integrate default global transformations
        first_global_transform = lambda x: x  # noqa: E731
        second_global_transform = lambda x: x  # noqa: E731
        super().__init__(transforms=[first_global_transform, second_global_transform])
