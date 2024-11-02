from typing import TypeAlias

import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class KNNEvaluator:
    def __init__(self, eval_loader: DataLoader[Batch], train_loader: DataLoader[Batch], model: torch.nn.Module) -> None:
        self.eval_loader = eval_loader
        self.train_loader = train_loader
        self.model = model

    @torch.no_grad()
    def evaluate(self, k: int = 20) -> float:
        self.model.eval()
        train_features, train_targets = self._extract_features(self.train_loader)
        eval_features, eval_targets = self._extract_features(self.eval_loader)
        knn = KNeighborsClassifier(n_neighbors=k)
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
