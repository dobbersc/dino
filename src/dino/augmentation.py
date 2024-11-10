from collections.abc import Callable, Sequence
from typing import TypeAlias

from torch import Tensor, nn
from torchvision.transforms import v2

import torch

Transform: TypeAlias = Callable[[Tensor], Tensor]


class Augmenter(nn.Module):
    """Utility class that applies a series of augmentation transforms to an input image."""

    def __init__(self, transforms: Transform | Sequence[Transform], repeats: int | Sequence[int] = 1) -> None:
        """Initializes an Augmenter.

        :param transforms: A single or multiple transforms to be applied to an image.
        :param repeats: The number of times to apply each transform.
            If an integer is specified, all transformations will be repeated the same number of times. Defaults to 1.
        """
        super().__init__()

        self.transforms = transforms if isinstance(transforms, Sequence) else (transforms,)
        self.repeats = repeats if isinstance(repeats, Sequence) else (repeats,) * len(self.transforms)

        if len(self.transforms) != len(self.repeats):
            msg: str = (
                f"Mismatch between number of transforms ({len(self.transforms)}) and "
                f"number of their repeats ({len(self.repeats)}). "
                "They should either be equal or the repeats argument must be an integer."
            )
            raise ValueError(msg)

        if any(n < 1 for n in self.repeats):
            msg = "No transformation can be repeated a non-positive number of times."
            raise ValueError(msg)

    def forward(self, image: Tensor) -> list[Tensor]:
        """Applies the sequence of transformations to the input image the specified number of times.

        :param image: The input image. Shape: [#channels, height, width].
        :return: A list containing the transformed images. Each of shape [#channels, modified_height, modified_width],
            where the modified height and width may also vary per image.
        """
        return [
            transform(image)
            for transform, repeat in zip(self.transforms, self.repeats, strict=True)
            for _ in range(repeat)
        ]


class DefaultLocalAugmenter(Augmenter):
    """Default Augmenter for local view augmentations of the DINO paper."""

    def __init__(
        self,
        repeats: int = 8,
        size: int | tuple[float, float] = 96,
        scale: float | tuple[float, float] = (0.05, 0.4),
    ) -> None:
        """Initializes an DefaultLocalAugmenter.

        :param repeats: The number of times the local view transform will be applied. Defaults to 8.
        :param size: The size (height and width) of the transformed image. Defaults to 96.
        :param scale: Specifies the lower and upper bounds for the random area of the crop, before resizing.
            The scale is defined with respect to the area of the original image. Defaults to (0.05, 0.4).
        """
        local_view_transform: Transform = v2.Compose(
            [
                v2.RandomResizedCrop(size=size, scale=scale, interpolation=v2.InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    transforms=[v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0])], p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ],
        )
        super().__init__(transforms=local_view_transform, repeats=repeats)


class DefaultGlobalAugmenter(Augmenter):
    """Default Augmenter for global view augmentations of the DINO paper."""

    def __init__(self, size: int | tuple[float, float] = 224, scale: float | tuple[float, float] = (0.4, 1.0)) -> None:
        """Initializes an DefaultGlobalAugmenter.

        :param size: The size (height and width) of the transformed image. Defaults to 224.
        :param scale: Specifies the lower and upper bounds for the random area of the crop, before resizing.
            The scale is defined with respect to the area of the original image. Defaults to (0.4, 1.0).
        """
        global_view_base_transform: Transform = v2.Compose(
            [
                v2.RandomResizedCrop(size=size, scale=scale, interpolation=v2.InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    transforms=[v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ],
        )

        normalize: Transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        first_global_view_transform: Transform = v2.Compose(
            [global_view_base_transform, v2.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0]), normalize],
        )

        second_global_view_transform: Transform = v2.Compose(
            [
                global_view_base_transform,
                v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0])], p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
                normalize,
            ],
        )

        super().__init__(transforms=[first_global_view_transform, second_global_view_transform])
