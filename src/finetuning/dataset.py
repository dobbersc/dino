import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class ImageNetDirectoryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Iterate over directories (each representing a class) in the main data directory
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            # Ensure it's a directory (ignore non-directory files)
            if os.path.isdir(class_path):
                class_name = class_dir  # Use the directory name as the class label
                if class_name.endswith(".tar"):
                    class_name = class_name[:-4]
                # Collect each image file in the class directory
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.JPEG'):  # Filter for image files
                        file_path = os.path.join(class_path, file_name)
                        # Store (image_path, class_label)
                        self.samples.append((file_path, class_name))

        # Create a mapping from class name to integer label
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(set([s[1] for s in self.samples]))}
        self.samples = [(s[0], self.class_to_idx[s[1]]) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Open image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label
