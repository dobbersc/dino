import argparse
import random
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import v2

from dino.augmentation import DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.datasets import ImageNetDirectoryDataset, unnorm

DISPLAY_NUM_IMAGES: int = 12
DISPLAY_ROWS: int = 2
DISPLAY_COLS: int = 6


# TODO: save figure to file
def display_data(dataset: ImageNetDirectoryDataset, class_idx=None, predict=None):
    # Assuming `dataset.get_image_by_class` is a generator
    max_length = 10
    if class_idx is not None:
        raw_images = list(islice(dataset.get_image_by_class(class_idx), DISPLAY_NUM_IMAGES))
        class_indices = [class_idx] * DISPLAY_NUM_IMAGES
        titles = None
    else:
        # pick random images
        img_indices = random.sample(range(len(dataset)), DISPLAY_NUM_IMAGES)
        raw_images, class_indices = zip(*[dataset[i] for i in img_indices], strict=False)
        titles = [f"g={dataset.get_class_name(g)[:max_length]}" for g in class_indices]
        print(titles)

    if predict is not None:
        # Assuming `predict` is a function that takes in a list of images and returns a list of predictions
        predictions = predict(raw_images)
        titles = [
            f"y={dataset.get_class_name(y)[:max_length]}\ng={dataset.get_class_name(g)[:max_length]}"
            for y, g in zip(predictions, class_indices, strict=False)
        ]

    images = [unnorm(img) for img in raw_images]

    # display 4x4
    fig, ax = plt.subplots(DISPLAY_ROWS, DISPLAY_COLS, figsize=(12, 4))
    for i, (img, ax) in enumerate(zip(images, ax.flatten(), strict=False)):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])

    if titles is None:
        title = dataset.get_class_name(class_idx)
        fig.suptitle(title)
    plt.show()


def _normalize_img(img: torch.Tensor) -> torch.Tensor:
    img = img.clone()
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    return img.to(torch.uint8)


def plot_original_image(image: torch.Tensor, output_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(_normalize_img(image.permute(1, 2, 0)).numpy())
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(output_dir / "original_image.pdf")


def plot_augmentations(image: torch.Tensor, output_dir: Path) -> None:
    repeats = 8
    global_augmenter = DefaultGlobalAugmenter()
    local_augmenter = DefaultLocalAugmenter(repeats=repeats)
    global_views = global_augmenter(image)
    local_views = local_augmenter(image)
    fig, axs = plt.subplots(nrows=3, ncols=4)
    flattened_axs = axs.reshape(-1)
    for i, gv in enumerate(global_views):
        flattened_axs[i + 1].imshow(_normalize_img(gv.permute(1, 2, 0)).numpy())
        flattened_axs[i + 1].axis("off")
        flattened_axs[i + 1].set_title(f"Global View {i+1}")
    fig.delaxes(flattened_axs[0])
    fig.delaxes(flattened_axs[3])

    for i, lv in enumerate(local_views):
        flattened_axs[i + 4].imshow(_normalize_img(lv.permute(1, 2, 0)).numpy())
        flattened_axs[i + 4].axis("off")
        flattened_axs[i + 4].set_title(f"Local View {i+1}")
    fig.tight_layout()
    plt.savefig(output_dir / "augmentations.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO visualizations.")
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True, help="Directory to save the results.",
    )
    parser.add_argument(
        "--image-path", "-i", type=str, required=True, help="Path to the input image.",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ],
    )
    tensor_image = transform(img)
    plot_original_image(tensor_image, Path(args.output_dir))
    plot_augmentations(tensor_image, Path(args.output_dir))


if __name__ == "__main__":
    main()
