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

        self.train_loader = DataLoader(
            TensorDataset(self.train_input_tensors, self.train_labels),
            batch_size=2,
        )

        self.val_loader_1 = DataLoader(
            TensorDataset(
                self.val_input_tensors,
                torch.Tensor([2, 3, 3]),  # expecting 33.33% accuracy
            ),
            batch_size=1,
        )

        self.val_loader_2 = DataLoader(
            TensorDataset(
                self.val_input_tensors,
                torch.Tensor([1, 2, 1]),  # expecting 66.66% accuracy
            ),
            batch_size=1,
        )

        self.val_loader_3 = DataLoader(
            TensorDataset(
                self.val_input_tensors,
                torch.Tensor([1, 2, 3]),  # expecting 100% accuracy
            ),
            batch_size=1,
        )

        self.model = self._MockModel()

    def test_knn_evaluator(self):
        evaluator = KNNEvaluator(self.val_loader_1, self.train_loader, self.model)
        assert evaluator.evaluate(k=1) == approx(0.33333, rel=1e-3)

        evaluator = KNNEvaluator(self.val_loader_2, self.train_loader, self.model)
        assert evaluator.evaluate(k=1) == approx(0.66666, rel=1e-3)

        evaluator = KNNEvaluator(self.val_loader_3, self.train_loader, self.model)
        assert evaluator.evaluate(k=1) == approx(1, rel=1e-3)
