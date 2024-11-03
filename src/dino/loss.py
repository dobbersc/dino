from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class DistillationLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        pass


class DINOLoss(DistillationLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        # Include centering + sharpening here
        # Note: In original implementation the centering operation is handled as an exponential moving average
        return torch.tensor((1,))
