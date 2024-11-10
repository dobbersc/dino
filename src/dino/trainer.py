import itertools
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from dino.augmentation import Augmenter, DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.datasets import ViewDataset, Views
from dino.loss import DINOLoss, DistillationLoss
from dino.utils.schedulers import CosineScheduler, Scheduler
from dino.utils.torch import get_module_device


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

    def _train_epoch(
        self,
        views_data_loader: DataLoader[Views],
        optimizer: Optimizer,
        loss_function: DistillationLoss,
        device: torch.device,
        teacher_scheduler: Scheduler,
    ) -> None:
        self.student.train()

        views: Views
        for views in views_data_loader:
            local_views: list[Tensor] = [local_view.to(device) for local_view in views.local_views]
            global_views: list[Tensor] = [global_view.to(device) for global_view in views.global_views]

            optimizer.zero_grad()

            student_output: Tensor = self._multi_forward(self.student, local_views + global_views)
            with torch.no_grad():
                # TODO: Check if using no_grad is different to disabling gradients of the teacher model's parameters
                teacher_output: Tensor = self._multi_forward(self.teacher, global_views)

            loss: Tensor = loss_function(student_output, teacher_output)

            loss.backward()  # TODO: Gradient clipping?
            optimizer.step()

            with torch.no_grad():
                momentum = teacher_scheduler.get_value()
                teacher_parameters: dict[str, torch.Tensor] = dict(self.teacher.named_parameters())
                student_parameters: dict[str, torch.Tensor] = dict(self.student.named_parameters())

                for param in student_parameters:
                    teacher_parameters[param].copy_(
                        momentum * teacher_parameters[param] + (1 - momentum) * student_parameters[param],
                    )
                teacher_scheduler.step()

    def train(
        self,
        max_epochs: int,
        batch_size: int,
        loss_function_class: type[DistillationLoss] = DINOLoss,
        loss_function_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[Optimizer] = SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
        teacher_network_momentum: tuple[float, float] = (0.996, 1.0),
        num_workers: int = 0,
        device: torch.device | None = None,
    ) -> None:
        device = get_module_device(self.student) if device is None else device

        self.student.to(device)
        self.teacher.to(device)

        loss_function: DistillationLoss = loss_function_class(**(loss_function_kwargs or {})).to(device)
        optimizer: Optimizer = optimizer_class(self.student.parameters(), **(optimizer_kwargs or {}))

        views_data_loader: DataLoader[Views] = DataLoader(
            self.view_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        teacher_scheduler: Scheduler = CosineScheduler(*teacher_network_momentum, max_epochs, len(views_data_loader))

        # TODO: Add logging
        for _ in range(1, max_epochs + 1):
            self._train_epoch(
                views_data_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                device=device,
                teacher_scheduler=teacher_scheduler,
            )
