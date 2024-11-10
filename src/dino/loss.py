import itertools
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class DistillationLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        pass


class DINOLoss(DistillationLoss):
    def __init__(
        self,
        output_size: int,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,  # TODO: Accept linear warm-up scheduler for temperatures
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()

        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum

        self.center: Tensor
        self.register_buffer("center", torch.zeros(1, output_size))

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """TODO: Write docstring.

        Args:
            teacher_output: ... Shape: [batch_size, #student_views, output_size].

        Returns:

        """
        batch_center: Tensor = teacher_output.mean(dim=(0, 1)).unsqueeze(dim=0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        """TODO: Write docstring.

        Args:
            student_output: ... Shape: [batch_size, #student_views, output_size].
            teacher_output: ... Shape: [batch_size, #teacher_views, output_size].

        Returns:
             The loss as a scalar tensor.
        """
        student_log_probs: Tensor = (student_output / self.student_temperature).log_softmax(dim=-1)
        teacher_probs: Tensor = ((teacher_output - self.center) / self.teacher_temperature).softmax(dim=-1).detach()

        num_loss_terms: int = 0
        average_loss: Tensor = torch.tensor(0, dtype=student_output.dtype)

        student_view_log_probs: Tensor  # Shape: [batch_size, output_size]
        teacher_view_probs: Tensor  # Shape: [batch_size, output_size]

        for (student_idx, student_view_log_probs), (teacher_idx, teacher_view_probs) in itertools.product(
            enumerate(student_log_probs.transpose(0, 1)),
            enumerate(teacher_probs.transpose(0, 1)),
        ):
            if student_idx == teacher_idx:
                continue

            loss: Tensor = (-teacher_view_probs * student_view_log_probs).sum(dim=-1)  # Shape: [batch_size]
            average_loss += loss.mean()
            num_loss_terms += 1

        average_loss /= num_loss_terms

        self.update_center(teacher_output)
        return average_loss
