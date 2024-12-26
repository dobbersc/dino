import itertools
import logging
import time
import typing
from collections import defaultdict
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader, Dataset

from dino import LOG_SEPARATOR
from dino.augmentation import Augmenter, DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.datasets import ViewDataset, Views
from dino.loss import DINOLoss, DistillationLoss
from dino.utils.logging import mlflow_log_metrics
from dino.utils.schedulers import ConstantScheduler, CosineScheduler, Scheduler
from dino.utils.torch import get_module_device

logger: logging.Logger = logging.getLogger(__name__)


@typing.final
class _Default:
    pass


_DEFAULT: Final[_Default] = _Default()


class DINOTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        view_dataset: Dataset[Tensor],
        local_augmenter: Augmenter | None = None,
        global_augmenter: Augmenter | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher

        self.view_dataset: ViewDataset = ViewDataset(
            view_dataset,
            local_augmenter=DefaultLocalAugmenter() if local_augmenter is None else local_augmenter,
            global_augmenter=DefaultGlobalAugmenter() if global_augmenter is None else global_augmenter,
        )

    def _log_parameters(self, max_epochs: int, batch_size: int, num_workers: int, device: torch.device) -> None:
        logger.info(LOG_SEPARATOR)
        logger.info("Training Model")
        logger.info(LOG_SEPARATOR)
        logger.info(self.student)

        # Log (hyper)parameters.
        logger.info(LOG_SEPARATOR)
        logger.info("Training Hyperparameters")
        logger.info(" - max_epochs: %r", max_epochs)
        logger.info(" - batch_size: %r", batch_size)

        logger.info(LOG_SEPARATOR)
        logger.info("Computational Parameters:")
        logger.info(" - num_workers: %r", num_workers)
        logger.info(" - device: %r", device)

        logger.info(LOG_SEPARATOR)
        logger.info("Dataset Information:")
        logger.info(" - train: %d data points", len(self.view_dataset))

    @staticmethod
    def multi_forward(model: nn.Module, views: list[Tensor]) -> Tensor:
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

        # Each output is of shape [len(views_group) * batch_size, output_size].
        # Note: The concatenation of the views in the respective group
        # organizes the batched view's dimensions as view first and batch second.
        outputs: list[Tensor] = [
            model(torch.cat(tuple(views_group), dim=0))
            for _, views_group in itertools.groupby(views, key=lambda view: view.size()[-2:])
        ]

        # Reshape outputs to [batch_size, #views, output_size].
        # To preserve the correctness of the views under the note above,
        # we first reshape to [#views, batch_size, output_size] and transpose afterwards.
        batch_size: int = views[0].size(dim=0)
        return torch.cat(outputs, dim=0).reshape(len(views), batch_size, -1).transpose(0, 1)

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
        lr_scheduler: LRScheduler | None,
        weight_decay_scheduler: Scheduler[float] | None,
        teacher_momentum_scheduler: Scheduler[float],
        max_gradient_norm: float,
        device: torch.device,
        # necessary to have consistent steps across epochs for mlflow logging:
        # I have the feeling @Conrad doesn't like this
        global_batch_index: int,
    ) -> tuple[float, dict[str, float]]:
        self.student.train()
        self.teacher.train()

        aggregated_loss: float = 0
        aggregated_inspection_metrics: defaultdict[str, float] = defaultdict(float)
        start_time: float = time.time()

        views: Views
        for batch_index, views in enumerate(data_loader, start=1):
            local_views: list[Tensor] = [local_view.to(device) for local_view in views.local_views]
            global_views: list[Tensor] = [global_view.to(device) for global_view in views.global_views]

            # Set weight decay if it is supported by the optimizer. We only regularize the first parameter group.
            weight_decay_is_supported: bool = "weight_decay" in optimizer.param_groups[0]
            if weight_decay_scheduler is not None and weight_decay_is_supported:
                optimizer.param_groups[0]["weight_decay"] = weight_decay_scheduler.get_value()

            optimizer.zero_grad()

            student_output: Tensor = self.multi_forward(self.student, global_views + local_views)
            with torch.no_grad():
                teacher_output: Tensor = self.multi_forward(self.teacher, global_views)

            loss, inspection_metrics = loss_function(student_output, teacher_output, compute_inspection_metrics=True)
            aggregated_loss += loss.item()
            for key, value in inspection_metrics.items():
                aggregated_inspection_metrics[key] += value.item()

            # Log intermediate results.
            if batch_index % max(1, len(data_loader) // 10) == 0:
                inspection_metrics_log: str = " ".join(
                    f"- {key} {value / batch_index:.4f}" for key, value in aggregated_inspection_metrics.items()
                )
                weight_decay_log: str = (
                    f"- weight decay {optimizer.param_groups[0]['weight_decay']:.4f}"
                    if weight_decay_is_supported
                    else ""
                )
                msg: str = (
                    f"batch {batch_index}/{len(data_loader)}"
                    f" - loss {aggregated_loss / batch_index:.4f}"
                    f" {inspection_metrics_log}"
                    f" - lr {optimizer.param_groups[0]['lr']:.8f}"
                    f" {weight_decay_log}"
                    f" - teacher momentum {teacher_momentum_scheduler.get_value():.4f}"
                    f" - time {time.time() - start_time:.2f}s"
                )
                logger.info(msg)

                mlflow_log_metrics(
                    "train_batch",
                    {
                        "loss": aggregated_loss / batch_index,
                        "lr": optimizer.param_groups[0]["lr"],
                        "teacher momentum": teacher_momentum_scheduler.get_value(),
                        **inspection_metrics,
                    },
                    global_batch_index,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=max_gradient_norm)

            optimizer.step()
            loss_function.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if weight_decay_scheduler is not None:
                weight_decay_scheduler.step()

            self._update_teacher(teacher_momentum_scheduler)
            global_batch_index += 1

        for key in aggregated_inspection_metrics:
            aggregated_inspection_metrics[key] /= len(data_loader)
        return aggregated_loss / len(data_loader), aggregated_inspection_metrics

    def train(
        self,
        max_epochs: int,
        batch_size: int,
        loss_function_class: type[DistillationLoss] = DINOLoss,
        loss_function_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[Optimizer] = SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_class: type[LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        weight_decay_scheduler: Scheduler[float] | _Default | None = _DEFAULT,
        teacher_momentum: float | Scheduler[float] | None = None,
        max_gradient_norm: float = 3.0,
        num_workers: int = 0,
        device: torch.device | None = None,
    ) -> None:
        device = get_module_device(self.student) if device is None else device

        self.student.to(device)
        self.teacher.to(device)

        # Initialize the data loader.
        data_loader: DataLoader[Views] = DataLoader(
            self.view_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        max_steps: int = max_epochs * len(data_loader)

        # Initialize the loss function.
        loss_function_kwargs = loss_function_kwargs or {}
        if loss_function_class is DINOLoss:
            loss_function_kwargs.setdefault("teacher_temperature", ConstantScheduler(0.04))  # TODO: Use right scheduler
        loss_function: DistillationLoss = loss_function_class(**loss_function_kwargs).to(device)

        # Initialize the optimizer.
        optimizer: Optimizer = optimizer_class(self.student.parameters(), **(optimizer_kwargs or {}))

        # Initialize the learning rate scheduler.
        lr_scheduler: LRScheduler | None = None
        lr_epoch_threshold: float = 25  # Set default lr scheduler if training for sufficient epochs.
        if lr_scheduler_class is None and max_epochs >= lr_epoch_threshold:
            milestone: int = 10 * len(data_loader) - 1  # Set milestone to 10 epochs.
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=1e-6, total_iters=milestone),
                    CosineAnnealingLR(optimizer, T_max=max_steps - milestone - 1, eta_min=1e-6),
                ],
                milestones=[milestone],
            )
        elif lr_scheduler_class is not None:
            lr_scheduler = lr_scheduler_class(optimizer, **(lr_scheduler_kwargs or {}))

        # Initialize the weight decay scheduler.
        if isinstance(weight_decay_scheduler, _Default):
            weight_decay_scheduler = CosineScheduler(max_steps=max_steps, initial=0.04, final=0.4)

        # Initialize the teacher momentum scheduler.
        if isinstance(teacher_momentum, float):
            teacher_momentum = ConstantScheduler(teacher_momentum)
        elif teacher_momentum is None:
            teacher_momentum = CosineScheduler(max_steps=max_steps, initial=0.996, final=1.0)

        # TODO: Log remaining parameters.
        self._log_parameters(
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        try:
            # Train the model.
            for epoch in range(1, max_epochs + 1):
                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d", epoch)

                loss, inspection_metrics = self._train_epoch(
                    data_loader,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    lr_scheduler=lr_scheduler,
                    weight_decay_scheduler=weight_decay_scheduler,
                    teacher_momentum_scheduler=teacher_momentum,
                    max_gradient_norm=max_gradient_norm,
                    device=device,
                    global_batch_index=(epoch - 1) * len(data_loader),
                )

                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d DONE", epoch)
                logger.info("Loss: %.8f", loss)
                for key, value in inspection_metrics.items():
                    name: str = "KL Divergence" if key == "kl_divergence" else key.replace("_", " ").title()
                    logger.info("%s: %.8f", name, value)

                mlflow_log_metrics(
                    "train_epoch",
                    {
                        "loss": loss,
                        **inspection_metrics,
                    },
                    epoch,
                )

        except KeyboardInterrupt:
            logger.info(LOG_SEPARATOR)
            logger.warning("Manually Interrupted Training!")

        logger.info(LOG_SEPARATOR)
        logger.info("Finished Training")
