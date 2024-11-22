import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import nn
from torch.utils.data import DataLoader

from dino.datasets import DatasetConfig, get_dataset
from dino.finetuning import FinetuningMode, finetune
from dino.models import BackboneConfig, HeadConfig, load_model_with_head
from dino.utils.random import set_seed

logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    model_dir: str = "./models"
    model_tag: str | None = None
    base_lr: float = MISSING
    backbone_lr: float = MISSING
    num_epochs: int = MISSING
    batch_size: int = MISSING
    mode: FinetuningMode = MISSING
    dataset: DatasetConfig = MISSING
    backbone: BackboneConfig = MISSING
    head: HeadConfig = MISSING


_cs = ConfigStore.instance()
_cs.store(
    group="finetune",
    name="base_finetune",
    node=FinetuningConfig,
)


def run_finetuning(cfg: FinetuningConfig) -> None:
    """Runs the finetuning process based on the specified configuration."""
    logger.info(f"Starting finetuning process... {cfg.model_tag=}")

    set_seed(42)

    dataset = get_dataset(cfg.dataset)

    msg = f"len(dataset): {len(dataset)}"
    logger.info(msg)
    # check if dataset has attribute num_classes
    output_dim: int = (
        dataset.num_classes if hasattr(dataset, "num_classes") else cfg.head.output_dim  # type: ignore[assignment]
    )
    msg = f"output_dim: {output_dim}"
    logger.info(msg)

    # TODO: implement exact experimental setup as in the paper

    model = load_model_with_head(
        model_type=cfg.backbone.model_type,
        head_type=cfg.head.model_type,
        output_dim=output_dim,
        hidden_dim=cfg.head.hidden_dim,
    )

    # TODO: implement checkpointing
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    msg = f"Using device: {device}"
    logger.info(msg)

    # TODO: implement validation
    validate = lambda _: None
    finetune(
        model,
        dataloader,
        criterion,
        mode=cfg.mode,
        base_lr=cfg.base_lr,
        backbone_lr=cfg.backbone_lr,
        num_epochs=cfg.num_epochs,
        device=device,
        validate=validate,
    )

    model_tag = cfg.model_tag or "finetuned"

    if cfg.mode == FinetuningMode.LINEAR_PROBE:
        model_tag += "_head"
        model_path = Path(cfg.model_dir) / model_tag
        model.save_head(model_path)
    elif cfg.mode == FinetuningMode.FULL_FINETUNE:
        model_tag += "_full"
        model_path = Path(cfg.model_dir) / model_tag
        model.save(model_path)
    else:
        msg = f"Invalid finetuning mode: {cfg.mode}"
        raise ValueError(msg)
