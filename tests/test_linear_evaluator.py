import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dino.evaluators import LinearEvaluator


class _ConstantProbabilisticClassificationModel(torch.nn.Module):
    def __init__(self, output_dim: int = 10, t: float = 1.0):
        super().__init__()
        self.output_dim = output_dim
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        softmax_input = torch.arange(self.output_dim, 0, step=-1) / self.t
        single_sample = torch.softmax(softmax_input, dim=-1)
        return single_sample.expand(batch_size, -1)


def create_loader(
    targets: list[int],
    target_sizes: list[int],
    input_dim: int,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    if len(targets) != len(target_sizes):
        msg = "Targets and target sizes length must match." f"Got {len(targets)} != {len(target_sizes)}."
        raise ValueError(msg)
    num_samples = sum(target_sizes)
    inputs = torch.rand(num_samples, input_dim)
    target_tensors = []
    for target, target_size in zip(targets, target_sizes, strict=False):
        target_tensors.append(torch.ones(target_size) * target)
    dataset = TensorDataset(inputs, torch.cat(target_tensors, dim=-1))
    return DataLoader(dataset, batch_size=batch_size)


@pytest.mark.parametrize(
    (
        "targets",
        "target_sizes",
        "output_dim",
        "topk",
        "expected",
    ),
    [
        ([0], [4], 5, (1,), {"top-1": 1.0}),
        ([0], [4], 5, (2,), {"top-2": 1.0}),
        ([1], [4], 5, (1,), {"top-1": 0.0}),
        ([0, 1], [4, 9], 5, (1,), {"top-1": 4 / 13}),
        ([0], [6], 5, (1, 2), {"top-1": 1.0, "top-2": 1.0}),
        ([1], [6], 5, (1, 2), {"top-1": 0.0, "top-2": 1.0}),
        ([2], [6], 5, (1, 2), {"top-1": 0.0, "top-2": 0.0}),
        ([0, 1], [2, 10], 5, (1, 2), {"top-1": 1 / 6, "top-2": 1.0}),
        ([0, 1], [2, 10], 5, (1, 2, 3), {"top-1": 1 / 6, "top-2": 1.0, "top-3": 1.0}),
        ([0, 2], [1, 10], 5, (1, 2), {"top-1": 1 / 11, "top-2": 1 / 11}),
        ([1, 2], [7, 3], 5, (1, 2), {"top-1": 0.0, "top-2": 0.7}),
        ([0, 1, 2], [2, 2, 5], 5, (1, 2), {"top-1": 2 / 9, "top-2": 4 / 9}),
        ([0, 1, 2], [5, 3, 2], 5, (1, 2), {"top-1": 0.5, "top-2": 0.8}),
        ([0, 1, 2, 3], [3, 3, 3, 3], 5, (3,), {"top-3": 9 / 12}),
        (list(range(10)), [10] * 10, 10, (5, 10), {"top-5": 0.5, "top-10": 1.0}),
    ],
)
def test_linear_evaluator(
    targets: list[int],
    target_sizes: list[int],
    output_dim: int,
    topk: tuple[int, ...],
    expected: dict[str, float],
) -> None:
    input_dim, batch_size = 5, 2
    model = _ConstantProbabilisticClassificationModel(output_dim)
    eval_loader = create_loader(
        targets=targets,
        target_sizes=target_sizes,
        input_dim=input_dim,
        batch_size=batch_size,
    )
    linear_evaluator = LinearEvaluator(eval_loader, model)  # type: ignore[arg-type]
    acc = linear_evaluator.evaluate(topk=topk)
    assert acc == expected
