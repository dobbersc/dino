import logging
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
import torch
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from tqdm import tqdm
from dino.utils.torch import detect_device

logger = logging.getLogger(__name__)
device = detect_device()

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        pass


Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class KNNEvaluator(Evaluator):
    def __init__(
        self,
        eval_loader: DataLoader[Batch],
        train_loader: DataLoader[Batch],
        model: torch.nn.Module,
    ) -> None:
        self.eval_loader = eval_loader
        self.train_loader = train_loader
        self.model = model.to(device)

    @torch.no_grad()
    def evaluate(
        self,
        k: int = 20,
    ) -> float:
        self.model.eval()
        train_features, train_targets = self._extract_features(self.train_loader)
        eval_features, eval_targets = self._extract_features(self.eval_loader)
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        logger.info("Fitting KNN model ...")
        knn.fit(train_features, train_targets)
        logger.info("Predicting ...")
        predictions = knn.predict(eval_features)
        logger.info("Calculating accuracy ...")
        return float(accuracy_score(eval_targets, predictions, normalize=True))

    def _extract_features(
        self,
        loader: DataLoader,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        features, targets = ([], [])
        logger.info("Extracting features ...")
        for images, targets_ in tqdm(loader):
            # Move images to the appropriate device
            images = images.to(device)
    
            # Forward pass to extract features
            output = self.model(images)
    
            # Move the output and targets to CPU and convert to NumPy
            features.extend(output.cpu().numpy())
            targets.extend(targets_.cpu().numpy())
        
        return features, targets


class LinearEvaluator(Evaluator):
    def __init__(
        self,
        eval_loader: DataLoader[Batch],
        model: torch.nn.Module,
    ) -> None:
        self.eval_loader = eval_loader
        self.model = model.to(device)

    @torch.no_grad()
    def evaluate(
        self,
        topk: tuple[int, ...] = (1,),
    ) -> dict[str, float]:
        
        self.model.eval()
        total_counts = np.zeros(len(topk))
        for images, targets in tqdm(self.eval_loader):
            images = images.to(device)
            targets = targets.to(device)
            output = self.model(images)
            correct_predictions = self._count_correct_predictions(output, targets, topk)
            total_counts = np.add(total_counts, correct_predictions)

        return {
            f"top-{topk[i]}": float(count / len(self.eval_loader.dataset))
            for i, count in enumerate(total_counts)
        }

    @staticmethod
    def _count_correct_predictions(
        output: torch.Tensor,
        targets: torch.Tensor,
        topk: tuple[int, ...],
    ) -> list[int]:
        maxk = max(topk)
        _, indices = output.topk(k=maxk, dim=-1, largest=True, sorted=True)
        expanded_targets = targets.unsqueeze(-1).expand_as(indices)
        return [int((torch.eq(indices[:, :k], expanded_targets[:, :k])).sum().item()) for k in topk]
