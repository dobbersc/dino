from pathlib import Path
from PIL import Image
import os
from torch.utils.data import Dataset
import random


class ImageNetDirectoryDataset(Dataset):
    def __init__(self, data_dir, transform=None, path_wnids=None, num_sample_classes=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Iterate over directories (each representing a wnid) in the main data directory
        for wnid_dir in os.listdir(data_dir):
            wnid_path = Path(data_dir) / wnid_dir
            # Ensure it's a directory (ignore non-directory files)
            if wnid_path.is_dir():
                # get the wnid name, the last part of the path
                wnid_name = wnid_path.parts[-1]

                if wnid_name.endswith(".tar"):
                    wnid_name = wnid_name[:-4]

                # check if there is another directory inside the wnid directory called images
                if (wnid_path / "images").is_dir():
                    wnid_path = wnid_path / "images"
                # Collect each image file in the wnid directory
                for file_name in os.listdir(wnid_path):
                    if file_name.endswith(".JPEG"):  # Filter for image files
                        file_path = wnid_path / file_name
                        self.samples.append((file_path, wnid_name))

        # Create a mapping from wnid name to integer label
        self.wnid_to_class_idx = {
            wnid_name: class_idx for class_idx, wnid_name in enumerate({s[1] for s in self.samples})
        }

        if num_sample_classes is not None:
            # smaple class_indices for the subset
            class_indices = random.sample(range(len(self.wnid_to_class_idx)), num_sample_classes)
            # filter samples for the subset
            self.samples = [
                s for s in self.samples if self.wnid_to_class_idx[s[1]] in class_indices
            ]
            # update the mapping
            self.wnid_to_class_idx = {
                wnid_name: class_idx
                for class_idx, wnid_name in enumerate({s[1] for s in self.samples})
            }

        self.class_idx_to_wnid = {v: k for k, v in self.wnid_to_class_idx.items()}
        self.samples = [(s[0], self.wnid_to_class_idx[s[1]]) for s in self.samples]

        if path_wnids is not None:
            self.class_idx_to_label = {}
            with open(path_wnids, "r") as f:
                for line in f:
                    wnid, label = line.strip().split("\t", 1)  # Split each line by tab
                    if wnid in self.wnid_to_class_idx:
                        class_idx = self.wnid_to_class_idx[wnid]
                        self.class_idx_to_label[class_idx] = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, class_idx):
        return self.class_idx_to_label.get(class_idx, None)

    def get_image_by_class(self, class_idx):
        for idx, s in enumerate(self.samples):
            if s[1] == class_idx:
                yield self.__getitem__(idx)[0]
