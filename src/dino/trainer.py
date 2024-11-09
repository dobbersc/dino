from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from dino.augmentation import Augmenter, DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.datasets import Views, ViewDataset
from dino.loss import DINOLoss, DistillationLoss
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

    def _train_epoch(
        self,
        views_data_loader: DataLoader[Views],
        optimizer: Optimizer,
        loss_function: DistillationLoss,
        device: torch.device,
    ) -> None:
        self.student.train()

        views: Views
        for views in views_data_loader:
            local_views: list[Tensor] = [local_view.to(device) for local_view in views.local_views]
            global_views: list[Tensor] = [global_view.to(device) for global_view in views.global_views]

            optimizer.zero_grad()

            student_output: Tensor = self.student(torch.cat((global_augmentations, local_augmentations), dim=1))
            with torch.no_grad():
                # TODO: Check if using no_grad is different to disabling gradients of the teacher model's parameters
                teacher_output: Tensor = self.teacher(global_augmentations)

            loss: Tensor = loss_function(student_output, teacher_output)

            loss.backward()  # TODO: Gradient clipping?
            optimizer.step()

            with torch.no_grad():
                ...  # TODO: Update teacher parameters with EMA

    def train(
        self,
        max_epochs: int,
        batch_size: int,
        loss_function_class: type[DistillationLoss] = DINOLoss,
        loss_function_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[Optimizer] = SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
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

        # TODO: Add logging
        for _ in range(1, max_epochs + 1):
            self._train_epoch(views_data_loader, optimizer=optimizer, loss_function=loss_function, device=device)
