import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nltk  # type: ignore[import-untyped]
import torch
from nltk.corpus import wordnet  # type: ignore[import-untyped]
from PIL.Image import Image
from tqdm import tqdm

from dino.datasets import ImageNetDirectoryDataset

random.seed(42)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def dataset_composition(
    dataset: ImageNetDirectoryDataset,
    dataset_path: str,
    dataset_name: str = "dataset",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if len(dataset) == 0:
        msg = "The dataset is empty."
        raise ValueError(msg)
    width_counter: Counter[int] = Counter()
    height_counter: Counter[int] = Counter()
    class_frequency_counter: Counter[str] = Counter()
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        if isinstance(img, Image):
            width, height = img.size
        elif isinstance(img, torch.Tensor):
            _, height, width = img.shape
        else:
            msg = "Specified image type is not supported."  # type: ignore[unreachable]
            raise TypeError(msg)
        width_counter[width] += 1
        height_counter[height] += 1
        class_frequency_counter[dataset.class_idx_to_wnid[label]] += 1

    avg_width = round(sum(width_counter.elements()) / width_counter.total())
    avg_height = round(sum(height_counter.elements()) / height_counter.total())
    composition = {
        "dataset_path": dataset_path,
        "average_image_size": {
            "width": avg_width,
            "height": avg_height,
        },
        "class_frequency": dict(class_frequency_counter),
        "size": len(dataset),
    }
    if output_dir is not None:
        with Path.open(output_dir / f"{dataset_name}_composition.json", "w", encoding="utf-8") as f:
            json.dump(composition, f, ensure_ascii=False, indent=4)
    return composition


def visualize_imagenet_classes(
    train_split: ImageNetDirectoryDataset,
    rows: int = 4,
    cols: int = 4,
    dataset_name: str = "dataset",
    output_path: Path | None = None,
) -> None:
    num_samples = rows * cols
    if train_split.num_classes < num_samples:
        class_idx = list(range(train_split.num_classes))
    else:
        class_idx = random.sample(range(train_split.num_classes), num_samples)

    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    for i, ax in enumerate(axs.reshape(-1)):
        if i < len(class_idx):
            # plot image
            ax.imshow(next(train_split.get_image_by_class(class_idx[i])))
            ax.axis("off")
            # plot the lemma of the synset as the title
            wnid = train_split.class_idx_to_wnid[class_idx[i]]
            offset = re.search(r"\d+", wnid)
            if offset is None:
                msg = "Wordnet identifier should be in part-of-speech with offset format"
                raise ValueError(msg)
            synset = wordnet.synset_from_pos_and_offset("n", int(offset.group()))
            lemma_name = synset.name().split(".")[0]
            ax.set_title(f"{lemma_name.replace('_', ' ').capitalize()}")
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path / f"{dataset_name}_samples.pdf")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet dataset analysis.")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        dest="dataset_name",
        help="Name of the dataset variant used as a prefix in artefacts.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True, help="Directory to save the results.",
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        required=True,
        help="Path to the train split of the dataset.",
    )
    parser.add_argument(
        "--rows",
        "-r",
        type=int,
        default=4,
        help="Specifies the number of rows in the plot containing the samples.",
    )
    parser.add_argument(
        "--cols",
        "-c",
        type=int,
        default=4,
        help="Specifies the number of columns in the plot containing the samples.",
    )

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_set = ImageNetDirectoryDataset(data_dir=Path(args.dataset_path))

    _ = dataset_composition(train_set, args.dataset_path, args.dataset_name, Path(args.output_dir))
    visualize_imagenet_classes(
        train_set, args.rows, args.cols, args.dataset_name, Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
