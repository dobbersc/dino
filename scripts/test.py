import os

from torch.utils.data import DataLoader
from torchvision import transforms

from dino import config
from dino.datasets import ImageNetDirectoryDataset
from dino.utils import list_directory_contents


def list_directory_contents(directory_path):
    directory_path = str(directory_path)
    for root, dirs, files in os.walk(directory_path):
        # Calculate the depth of the current directory to format the output
        depth = root.replace(directory_path, "").count(os.sep)
        indent = " " * 4 * depth
        print(f"{indent}{os.path.basename(root)}/")

        # List directories
        sub_indent = " " * 4 * (depth + 1)
        for d in dirs:
            print(f"{sub_indent}{d}/")

        # List files
        for f in files:
            print(f"{sub_indent}{f}")


if __name__ == "__main__":
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for ImageNet models
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )

    # print dataset dir
    list_directory_contents(config.IMAGENET_DIR)

    # # Create the dataset and DataLoader
    dataset = ImageNetDirectoryDataset(
        config.IMAGENET_DIR, transform=transform, num_sample_classes=3
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example: iterate over DataLoader
    for images, labels in dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
