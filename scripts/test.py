import finetuning
from finetuning.dataset import ImageNetDirectoryDataset
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    print(finetuning.__version__)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for ImageNet models
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ]
    )

    # Create the dataset and DataLoader
    data_dir = "/input-data"
    dataset = ImageNetDirectoryDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example: iterate over DataLoader
    for images, labels in dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
