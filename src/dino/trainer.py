import itertools
import logging
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from dino import LOG_SEPARATOR
from dino.augmentation import Augmenter, DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.datasets import ViewDataset, Views
from dino.loss import DINOLoss, DistillationLoss
from dino.utils.schedulers import ConstantScheduler, CosineScheduler, Scheduler
from dino.utils.torch import get_module_device

logger: logging.Logger = logging.getLogger(__name__)


class DINOTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataset: Dataset[Tensor],
        local_augmenter: Augmenter | None = None,
        global_augmenter: Augmenter | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher

        self.view_dataset: ViewDataset = ViewDataset(
            dataset,
            local_augmenter=DefaultLocalAugmenter() if local_augmenter is None else local_augmenter,
            global_augmenter=DefaultGlobalAugmenter() if global_augmenter is None else global_augmenter,
        )

    @staticmethod
    def _multi_forward(model: nn.Module, views: list[Tensor]) -> Tensor:
        """Performs a forward pass separately for each consecutive group of view resolutions.

        Since this function forwards each consecutive group of view resolutions,
        the order of the model's outputs corresponding to inputs is preserved.

        Args:
            model: The model to forward the views through.
            views: A list of image views. Each view at index `i` of shape:
                [batch_size, #channels, height_i, width_i], where height_i and width_i may vary per view.

        Returns:
            The model's outputs for each view. Shape: [batch_size, #views, output_size].
        """
        if not views:
            msg: str = "The 'views' list must contain at least one view for the forward pass."
            raise ValueError(msg)

        # Each output is of shape [batch_size * len(views_group), output_size]
        outputs: list[Tensor] = [
            model(torch.cat(tuple(views_group), dim=0))
            for _, views_group in itertools.groupby(views, key=lambda view: view.size()[-2:])
        ]

        # Reshape outputs to [batch_size, #views, output_size]
        batch_size: int = views[0].size(dim=0)
        return torch.cat(outputs, dim=0).reshape(batch_size, len(views), -1)

    @torch.no_grad()
    def _update_teacher(self, teacher_momentum_scheduler: Scheduler[float]) -> None:
        """...

        Note: Updates all parameters in the teacher that are also available in the student.

        Args:
            teacher_momentum_scheduler: ...

        Returns:
            ...
        """
        momentum: float = teacher_momentum_scheduler.get_value()
        teacher_parameters: dict[str, torch.Tensor] = dict(self.teacher.named_parameters())

        for name, student_parameter in self.student.named_parameters():
            teacher_parameters[name].copy_(
                momentum * teacher_parameters[name] + (1 - momentum) * student_parameter,
            )

        teacher_momentum_scheduler.step()

    def _train_epoch(
        self,
        data_loader: DataLoader[Views],
        optimizer: Optimizer,
        loss_function: DistillationLoss,
        teacher_momentum_scheduler: Scheduler[float],
        device: torch.device,
    ) -> float:
        self.student.train()
        self.teacher.train()

        aggregated_loss: float = 0
        start_time: float = time.time()

        views: Views
        for batch_index, views in enumerate(data_loader, start=1):
            local_views: list[Tensor] = [local_view.to(device) for local_view in views.local_views]
            global_views: list[Tensor] = [global_view.to(device) for global_view in views.global_views]

            optimizer.zero_grad()

            student_output: Tensor = self._multi_forward(self.student, global_views + local_views)
            with torch.no_grad():
                teacher_output: Tensor = self._multi_forward(self.teacher, global_views)

            loss: Tensor = loss_function(student_output, teacher_output)
            aggregated_loss += loss.item()

            # Log intermediate results.
            if batch_index % max(1, len(data_loader) // 10) == 0:
                msg: str = (
                    f"batch {batch_index}/{len(data_loader)}"
                    f" - loss {aggregated_loss / batch_index:.8f}"
                    f" - lr {optimizer.param_groups[0]['lr']:.8f}"
                    f" - teacher momentum {teacher_momentum_scheduler.get_value():.8f}"
                    f" - time {time.time() - start_time:.2f}s"
                )
                logger.info(msg)

            loss.backward()  # TODO: Gradient clipping?
            optimizer.step()
            loss_function.step()

            self._update_teacher(teacher_momentum_scheduler)

        return aggregated_loss / len(data_loader)

    def train(
        self,
        max_epochs: int,
        batch_size: int,
        loss_function_class: type[DistillationLoss] = DINOLoss,
        loss_function_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[Optimizer] = SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
        teacher_momentum: float | Scheduler[float] | None = None,
        num_workers: int = 0,
        device: torch.device | None = None,
    ) -> None:
        device = get_module_device(self.student) if device is None else device

        self.student.to(device)
        self.teacher.to(device)

        views_data_loader: DataLoader[Views] = DataLoader(
            self.view_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        loss_function_kwargs = loss_function_kwargs or {}
        if loss_function_class is DINOLoss:
            loss_function_kwargs.setdefault("teacher_temperature", ConstantScheduler(0.04))  # TODO: Use right scheduler
        loss_function: DistillationLoss = loss_function_class(**loss_function_kwargs).to(device)

        optimizer: Optimizer = optimizer_class(self.student.parameters(), **(optimizer_kwargs or {}))

        max_steps: int = max_epochs * len(views_data_loader)
        if isinstance(teacher_momentum, float):
            teacher_momentum = ConstantScheduler(teacher_momentum)
        elif teacher_momentum is None:
            teacher_momentum = CosineScheduler(max_steps=max_steps, initial=0.996, final=1.0)

        try:
            for epoch in range(1, max_epochs + 1):
                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d", epoch)

                loss: float = self._train_epoch(
                    views_data_loader,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    teacher_momentum_scheduler=teacher_momentum,
                    device=device,
                )

                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d DONE", epoch)
                logger.info("Loss: %.8f", loss)

        except KeyboardInterrupt:
            logger.info(LOG_SEPARATOR)
            logger.warning("Manually Interrupted Training!")

        logger.info(LOG_SEPARATOR)
        logger.info("Finished Training")
