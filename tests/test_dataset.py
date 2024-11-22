import unittest
from collections import Counter

from torchvision.datasets import ImageFolder

from dino.datasets import ImageNetDirectoryDataset, split_imagefolder


class TestImageNetDirectoryDataset(unittest.TestCase):
    def setUp(self):
        # Provide the path to the dataset
        self.path_to_data = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/train"  # Update with the correct path
        self.split_ratio = 0.9
        # Initialize the train and validation datasets
        self.train_dataset = ImageNetDirectoryDataset(
            data_dir=self.path_to_data,
            train=True,
            split_ratio=self.split_ratio,
        )
        self.val_dataset = ImageNetDirectoryDataset(
            data_dir=self.path_to_data,
            train=False,
            split_ratio=self.split_ratio,
        )

    def test_train_val_split_contains_all_classes(self):
        # Get class indices for train and validation datasets
        train_classes = [label for _, label in self.train_dataset.samples]
        val_classes = [label for _, label in self.val_dataset.samples]

        # Ensure all classes are present in both splits
        train_classes_set = set(train_classes)
        val_classes_set = set(val_classes)
        all_classes = set(self.train_dataset.wnid_to_class_idx.values()).union(
            self.val_dataset.wnid_to_class_idx.values(),
        )  # From the dataset

        print("All classes: ", all_classes)
        assert (
            train_classes_set == all_classes
        ), "Not all classes are present in the training dataset."
        assert (
            val_classes_set == all_classes
        ), "Not all classes are present in the validation dataset."

        # Ensure each class has at least one sample in both splits
        train_counts = Counter(train_classes)
        val_counts = Counter(val_classes)

        for class_idx in all_classes:
            assert (
                train_counts[class_idx] > 0
            ), f"Class {class_idx} has no samples in the training split."
            assert (
                val_counts[class_idx] > 0
            ), f"Class {class_idx} has no samples in the validation split."

    def test_train_val_split_exclusiveness(self):
        # Ensure there is no overlap between the train and validation datasets
        train_paths = [path for path, _ in self.train_dataset.samples]
        val_paths = [path for path, _ in self.val_dataset.samples]

        assert (
            len(set(train_paths).intersection(val_paths)) == 0
        ), "Train and validation datasets are not exclusive."


class TestSplitImageFolder(unittest.TestCase):
    def setUp(self):
        # Path to the dataset for testing
        self.path_to_data = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/train"  # Update with the correct path
        self.split_ratio = 0.9
        # Load the ImageFolder dataset
        self.dataset = ImageFolder(root=self.path_to_data)

        # Split the dataset into train and validation subsets
        self.train_dataset = split_imagefolder(
            self.dataset,
            split_ratio=self.split_ratio,
            train=True,
        )
        self.val_dataset = split_imagefolder(
            self.dataset,
            split_ratio=self.split_ratio,
            train=False,
        )

    def test_split_imagefolder_balanced_classes(self):
        # Verify that both splits contain all classes
        train_classes = [label for _, label in self.train_dataset]
        val_classes = [label for _, label in self.val_dataset]

        train_classes_set = set(train_classes)
        val_classes_set = set(val_classes)
        all_classes = set(self.dataset.class_to_idx.values())

        assert (
            train_classes_set == all_classes
        ), "Not all classes are present in the training split."
        assert (
            val_classes_set == all_classes
        ), "Not all classes are present in the validation split."

        # Ensure each class has at least one sample in both splits
        train_counts = Counter(train_classes)
        val_counts = Counter(val_classes)

        for class_idx in all_classes:
            assert (
                train_counts[class_idx] > 0
            ), f"Class {class_idx} has no samples in the training split."
            assert (
                val_counts[class_idx] > 0
            ), f"Class {class_idx} has no samples in the validation split."

    def test_split_imagefolder_correct_ratios(self):
        # Verify that the train/validation split ratio is correct
        total_samples = len(self.dataset)
        expected_train_size = int(total_samples * self.split_ratio)
        expected_val_size = total_samples - expected_train_size

        assert len(self.train_dataset) == expected_train_size, "Training split size is incorrect."
        assert len(self.val_dataset) == expected_val_size, "Validation split size is incorrect."

    def test_train_val_split_exclusiveness(self):
        # Ensure there is no overlap between the train and validation datasets
        train_paths = [path for path, _ in self.train_dataset.samples]
        val_paths = [path for path, _ in self.val_dataset.samples]

        assert (
            len(set(train_paths).intersection(val_paths)) == 0
        ), "Train and validation datasets are not exclusive."
