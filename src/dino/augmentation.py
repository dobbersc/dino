from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torchvision import transforms  # type: ignore[import-untyped]

Transform: TypeAlias = Callable[[Tensor], Tensor]


class Augmenter(nn.Module):
    """Utility class that applies a series of augmentation transforms to an input image."""

    def __init__(self, transforms: Transform | Sequence[Transform], repeats: int | Sequence[int] = 1) -> None:
        """Initializes an Augmenter.

        :param transforms: A single or multiple transforms to be applied to an image.
        :param repeats: The number of times to apply each transform.
                        If an integer is specified, all transformations will be repeated the same number of times,
                        defaults to 1.
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

    def forward(self, image: Tensor) -> Tensor:
        """Applies the list of transformations to the input image the specified number of times.

        :param image: Input image.
        :return: Stacked transformed images.
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
    def __init__(self, repeats: int = 8, size: int = 96, scale: tuple[float, float] = (0.05, 0.4)) -> None:
        """Initializes an DefaultLocalAugmenter.

        :param repeats: The number of times the local view transform will be applied, defaults to 8.
        :param size: Size of the transformed image, defaults to 96.
        :param scale: The area proportion of the original image to be included in the crop, defaults to (0.05, 0.4).
        """
        local_view_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    transforms=[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        super().__init__(transforms=local_view_transform, repeats=repeats)


class DefaultGlobalAugmenter(Augmenter):
    def __init__(self, size: int = 224, scale: tuple[float, float] = (0.4, 1.0)) -> None:
        """Initializes an DefaultGlobalAugmenter.

        :param size: Size of the transformed image, defaults to 224
        :param scale: The area proportion of the original image to be included in the crop, defaults to (0.4, 1.0)
        """
        global_view_base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    transforms=[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        first_global_view_transform = transforms.Compose(
            [global_view_base_transform, transforms.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0]), normalize]
        )

        second_global_view_transform = transforms.Compose(
            [
                global_view_base_transform,
                transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=23, sigma=[0.1, 2.0])], p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
                normalize,
            ]
        )

        super().__init__(transforms=[first_global_view_transform, second_global_view_transform])
