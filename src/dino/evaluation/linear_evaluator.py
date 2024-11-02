import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import TypeAlias

Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor]

class LinearEvaluator:
    def __init__(self, eval_loader: DataLoader[Batch], model: torch.nn.Module) -> None:
        self.eval_loader = eval_loader
        self.model = model

    @torch.no_grad()
    def evaluate(self, topk: tuple[int, ...] = (1,)) -> dict[str, float]:
        self.model.eval()
        total_counts = np.zeros(len(topk))
        for images, targets in self.eval_loader:
            output = self.model(images)
            correct_predictions = self._count_correct_predictions(output, targets, topk)
            total_counts = np.add(total_counts, correct_predictions)

        return {f"top-{topk[i]}": float(count / len(self.eval_loader)) for i, count in enumerate(total_counts)}

    @staticmethod
    def _count_correct_predictions(output: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...]) -> list[int]:
        maxk = max(topk)
        _, indices = output.topk(k=maxk, dim=-1, largest=True, sorted=True)
        expanded_targets = targets.unsqueeze(-1).expand_as(indices)
        return [int((torch.eq(indices[:, :k], expanded_targets[:, :k])).sum().item()) for k in topk]
