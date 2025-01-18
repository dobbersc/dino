import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from dino.datasets import ContrastiveLearningDataset
from dino.models import BackboneConfig, HeadConfig, HeadType, load_model_with_head
from dino.simclr import SimCLR
from dino.utils.torch import detect_device

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SimCLRConfig:
    data_dir: str = MISSING
    n_views: int = 2
    image_size: int = 96
    batch_size: int = 1
    num_workers: int = 1
    epochs: int = 100
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4
    temperature: float = 0.07
    fp16_precision: bool = False

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(
        default_factory=lambda: HeadConfig(
            model_type=HeadType.SIMCLR_HEAD,
            output_dim=128,
            hidden_dim=384,
        ),
    )

    model_dir: str = str(Path.cwd() / "models")
    model_tag: str | None = None


_cs = ConfigStore.instance()
_cs.store(name="base_simclr_config", node=SimCLRConfig)


@hydra.main(version_base=None, config_name="base_simclr_config")
def main(cfg: SimCLRConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    ds = ContrastiveLearningDataset(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        size=cfg.image_size,
    )

    device = detect_device()

    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
    )
    cfg.head.model_type = HeadType.SIMCLR_HEAD
    model = load_model_with_head(cfg.backbone, cfg.head)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=0,
        last_epoch=-1,
    )

    simclr = SimCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        temperature=cfg.temperature,
        n_views=cfg.n_views,
        fp16_precision=cfg.fp16_precision,
    )
    simclr.train(train_loader)

    model_name = cfg.model_tag or f"{cfg.backbone.model_type.value}_simclr"
    model.save_backbone(Path(cfg.model_dir) / model_name)
