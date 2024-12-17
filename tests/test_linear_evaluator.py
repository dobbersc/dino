import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dino.evaluators import LinearEvaluator


class _ConstantDeterministicClassificationModel(torch.nn.Module):
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        zeros = torch.zeros(batch_size, self.output_dim - 1)
        ones = torch.ones(batch_size, 1)
        return torch.cat((ones, zeros), dim=-1)


@pytest.fixture
def constant_model() -> _ConstantDeterministicClassificationModel:
    return _ConstantDeterministicClassificationModel()


def test_constant(constant_model: _ConstantDeterministicClassificationModel) -> None:
    num_samples = 10
    features = 5
    inputs = torch.rand(num_samples, features)
    targets = torch.zeros(num_samples)
    dataset = TensorDataset(inputs, targets)
    eval_loader = DataLoader(dataset, batch_size=2)
    linear_evaluator = LinearEvaluator(eval_loader, constant_model)  # type: ignore[arg-type]
    acc = linear_evaluator.evaluate(topk=(1,))
    assert acc["top-1"] == 1.0
