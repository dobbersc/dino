import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino.models import ModelWithHead

logger = logging.getLogger(__name__)


class FinetuningMode(Enum):
    """Enumeration for finetuning modes.

    Attributes:
        LINEAR_PROBE (str): Finetuning mode where only the head of the model is trained.
        FULL_FINETUNE (str): Finetuning mode where both the backbone and head of the model are trained.
    """

    LINEAR_PROBE = "linear_probe"
    FULL_FINETUNE = "full_finetune"


def get_optimizer(
    model: ModelWithHead,
    base_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    mode: FinetuningMode = FinetuningMode.LINEAR_PROBE,
) -> optim.Optimizer:
    """Returns the optimizer for finetuning the model based on the specified mode.

    Args:
        model: The model to be optimized, containing a backbone and a head.
        base_lr: The base learning rate for the optimizer. Default is 1e-3.
        backbone_lr: The learning rate for the backbone (only with FULL_FINETUNE). Default is 1e-4.
        mode: The mode of finetuning, either LINEAR_PROBE or FULL_FINETUNE.

    Returns:
        optim.Optimizer: The initialized optimizer for training the model.
    """
    optimizer_params: dict[str, Any] = {
        "momentum": 0.9,  # Use the same configuration as in the paper
        "weight_decay": 0,
    }
    match mode:
        case FinetuningMode.LINEAR_PROBE:
            optimizer_params |= {"lr": base_lr, "params": model.head_parameters()}
        case FinetuningMode.FULL_FINETUNE:
            # In the corresponding paper code,
            # they don't show the setup for full finetuning
            optimizer_params |= {
                "params": [
                    {"params": model.backbone_parameters(), "lr": backbone_lr},
                    {"params": model.head_parameters(), "lr": base_lr},
                ],
            }

    return optim.SGD(**optimizer_params)  # Use SGD optimizer as they do in the paper


def get_scheduler(optimizer: optim.Optimizer, n_epochs: int) -> optim.lr_scheduler.LRScheduler:
    """Returns the scheduler for the optimizer.

    Args:
        optimizer: The optimizer to be scheduled.
        n_epochs: The number of epochs to train the model.

    Returns:
        optim.lr_scheduler._LRScheduler: The scheduler for the optimizer.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=0)


# TODO: maybe implement load from model checkpoint
def train(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Trains the model using the specified dataloader, criterion, and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        dataloader (DataLoader): The dataloader providing the training data.
        criterion (nn.Module): The loss function used for optimization.
        optimizer (optim.Optimizer): The optimizer used for model parameter updates.
        device (str): The device to use for training, e.g., "cpu" or "cuda". Default is "cpu".
    """
    model.train()
    model.to(device)

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return {"loss": loss.item()}  # training stats

    # set optimizer


# Question: configurable optimizer or scheduler?
def finetune(
    model: ModelWithHead,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    base_lr: float,
    backbone_lr: float,
    mode: FinetuningMode,
    num_epochs: int,
    device: torch.device,
    validate: Callable[[nn.Module], Any] | None = None,
) -> None:
    """Performs finetuning on the model using the specified dataloader, criterion, and mode.

    Args:
        model: The model to be finetuned, containing a backbone and a head.
        dataloader: The dataloader providing the training data.
        criterion: The loss function used for optimization.
        base_lr: The base learning rate for the optimizer. Default is 1e-3.
        mode: The mode of finetuning, either LINEAR_PROBE or FULL_FINETUNE.
        num_epochs: The number of epochs to train the model. Default is 10.
        device: The device to use for training, e.g., "cpu" or "cuda". Default is "cpu".
        validate: The validation function to evaluate the model after each epoch.
    """
    if mode == FinetuningMode.LINEAR_PROBE:
        model.freeze_backbone()
    optimizer = get_optimizer(
        model,
        base_lr=base_lr,
        backbone_lr=backbone_lr,
        mode=mode,
    )
    scheduler = get_scheduler(optimizer, num_epochs)
    for epoch in range(num_epochs):
        train_stats = train(model, dataloader, criterion, optimizer, device=device)
        scheduler.step()
        if validate is not None:
            validate(model)
        msg = f"Epoch [{epoch+1}/{num_epochs}] - train loss: {train_stats['loss']}"
        logger.info(msg)
