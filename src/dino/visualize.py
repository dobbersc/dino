import random
from itertools import islice

import matplotlib.pyplot as plt

from dino.datasets import ImageNetDirectoryDataset, unnorm

DISPLAY_NUM_IMAGES: int = 12
DISPLAY_ROWS: int = 2
DISPLAY_COLS: int = 6


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

    print("len(raw_images):", len(raw_images))
    if predict is not None:
        # Assuming `predict` is a function that takes in a list of images and returns a list of predictions
        print("Predicting...")
        predictions = predict(raw_images)
        print("Predictions:", predictions)
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
