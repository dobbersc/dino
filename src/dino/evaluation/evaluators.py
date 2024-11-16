from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
import torch
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from torch.utils.data import DataLoader


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        pass


Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class KNNEvaluator(Evaluator):
    def __init__(self, eval_loader: DataLoader[Batch], train_loader: DataLoader[Batch], model: torch.nn.Module) -> None:
        self.eval_loader = eval_loader
        self.train_loader = train_loader
        self.model = model

    @torch.no_grad()
    def evaluate(self, k: int = 20) -> float:
        self.model.eval()
        train_features, train_targets = self._extract_features(self.train_loader)
        eval_features, eval_targets = self._extract_features(self.eval_loader)
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(train_features, train_targets)
        predictions = knn.predict(eval_features)
        return float(accuracy_score(eval_targets, predictions, normalize=True))

    def _extract_features(self, loader: DataLoader[Batch]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features, targets = ([], [])
        for images, targets_ in loader:
            output = self.model(images)
            features.extend(output)
            targets.extend(targets_)
        return features, targets


class LinearEvaluator(Evaluator):
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
