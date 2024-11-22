import unittest

import torch
from pytest import approx
from torch.utils.data import DataLoader, TensorDataset

from dino.evaluators import KNNEvaluator


class TestKnnEvaluator(unittest.TestCase):
    class _MockModel(torch.nn.Module):
        def forward(self, x):
            return x

    def setUp(self):
        # Create mock input image tensors
        self.train_input_tensors = torch.Tensor(
            [
                # class 1
                [1, 1, 1, 1],
                [1, 1, 1, 2],
                # class 2
                [2, 2, 2, 1],
                [2, 2, 2, 2],
                # class 3
                [3, 3, 3, 1],
                [3, 3, 3, 2],
            ],
        )

        self.train_labels = torch.Tensor([1, 1, 2, 2, 3, 3])

        self.val_input_tensors = torch.Tensor(
            [
                # class 1
                [1, 1, 1, 1],
                # class 2
                [2, 2, 2, 1],
                # class 3
                [3, 3, 3, 1],
            ],
        )

        self.val_labels = torch.Tensor([1, 2, 1])

        self.train_dataset = TensorDataset(self.train_input_tensors, self.train_labels)
        self.val_dataset = TensorDataset(self.val_input_tensors, self.val_labels)
        self.train_loader = DataLoader(self.train_dataset, batch_size=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)

        self.model = self._MockModel()

    def test_knn_evaluator(self):
        evaluator = KNNEvaluator(self.val_loader, self.train_loader, self.model)
        accuracy = evaluator.evaluate(k=1)

        assert accuracy == approx(0.66666, rel=1e-3)
