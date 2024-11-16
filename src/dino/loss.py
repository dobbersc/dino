import itertools
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from dino.utils.schedulers import ConstantScheduler, Scheduler


class DistillationLoss(nn.Module, ABC):
    """Abstract base class for losses in knowledge distillation frameworks."""

    @abstractmethod
    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        """Computes the distillation loss given the student and teacher model's outputs."""

    def step(self) -> None:
        """Informs the internal schedulers to take a step in the schedule."""


class DINOLoss(DistillationLoss):
    """Distillation loss specific to the DINO (Self-Distillation with No Labels) framework.

    TODO: Briefly describe the loss.
    """

    def __init__(
        self,
        output_size: int,
        student_temperature: float | Scheduler[float] = 0.1,
        teacher_temperature: float | Scheduler[float] = 0.04,
        center_momentum: float | Scheduler[float] = 0.9,
    ) -> None:
        """Initializes a DINOLoss.

        Args:
            output_size: The output size of the student and teacher model's logits
                (shape: [batch_size, #views, output_size]).
            student_temperature: The temperature for scaling the student's logits with a sharpened softmax.
                Supports constant and scheduled temperatures. Defaults to 0.1.
            teacher_temperature: The temperature for scaling the teacher's logits with a sharpened softmax.
                Supports constant and scheduled temperatures. Defaults to 0.04.
            center_momentum: The momentum for updating the moving average of teacher's logits centers.
                Support constant and scheduled momentums. Defaults to 0.9.
        """
        super().__init__()

        self.student_temperature = (
            ConstantScheduler(student_temperature) if isinstance(student_temperature, float) else student_temperature
        )
        self.teacher_temperature = (
            ConstantScheduler(teacher_temperature) if isinstance(teacher_temperature, float) else teacher_temperature
        )
        self.center_momentum = (
            ConstantScheduler(center_momentum) if isinstance(center_momentum, float) else center_momentum
        )

        self.center: Tensor
        self.register_buffer("center", torch.zeros(1, output_size))

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """Updates the center of the teacher model's logits.

        The center is an exponential moving average of the mean teacher's logits across all batches.

        Args:
            teacher_output: The teacher model's logits. Shape: [batch_size, #global_views, output_size].
        """
        center_momentum: float = self.center_momentum.get_value()

        batch_center: Tensor = teacher_output.mean(dim=(0, 1)).unsqueeze(dim=0)
        self.center = center_momentum * self.center + (1 - center_momentum) * batch_center

    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        """Computes the DINO loss.

        Args:
            student_output: The student model's logits. Shape: [batch_size, #global_views + #local_views, output_size].
                Must contain the global views as preceding elements to align with the teacher model's output.
            teacher_output: The teacher model's logits. Shape: [batch_size, #global_views, output_size].

        Returns:
            The averaged loss as a scalar tensor.
        """
        student_temperature: float = self.student_temperature.get_value()
        teacher_temperature: float = self.teacher_temperature.get_value()

        student_log_probs: Tensor = (student_output / student_temperature).log_softmax(dim=-1)
        teacher_probs: Tensor = ((teacher_output - self.center) / teacher_temperature).softmax(dim=-1).detach()

        num_loss_terms: int = 0
        average_loss: Tensor = torch.tensor(0, dtype=student_output.dtype, device=teacher_probs.device)

        student_view_log_probs: Tensor  # Shape: [batch_size, output_size]
        teacher_view_probs: Tensor  # Shape: [batch_size, output_size]
        for (student_idx, student_view_log_probs), (teacher_idx, teacher_view_probs) in itertools.product(
            enumerate(student_log_probs.transpose(0, 1)),
            enumerate(teacher_probs.transpose(0, 1)),
        ):
            if student_idx == teacher_idx:
                # Skip loss calculation when the student and teacher operated on an identical view.
                continue

            loss: Tensor = (-teacher_view_probs * student_view_log_probs).sum(dim=-1)  # Shape: [batch_size]
            average_loss += loss.mean()
            num_loss_terms += 1

        average_loss /= num_loss_terms

        self.update_center(teacher_output)
        return average_loss

    def step(self) -> None:
        self.student_temperature.step()
        self.teacher_temperature.step()
        self.center_momentum.step()
