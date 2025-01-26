import logging
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from dino.datasets import ContrastiveLearningDataset, val_transform
from dino.evaluators import KNNEvaluator
from dino.models import BackboneConfig, HeadConfig, HeadType, ModelWithHead, load_model_with_head
from dino.simclr import SimCLR
from dino.utils.logging import log_hydra_config_to_mlflow
from dino.utils.torch import detect_device

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    data_dir: str = MISSING
    train: str = "train"
    validation: str = "val"
    knn_k: int = 5
    batch_size: int = 256
    num_workers: int = 1


@dataclass
class SimCLRConfig:
    data_dir: str = MISSING
    n_views: int = 2
    image_size: int = 224
    batch_size: int = 64  # needs to be adjusted
    num_workers: int = 1
    epochs: int = 100
    learning_rate: float = 0.0003  # default as in the repo
    weight_decay: float = 1e-4  # default as in the repo
    temperature: float = 0.07  # default as in the repo
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
    experiment_tag: str = "Default"

    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)


_cs = ConfigStore.instance()
_cs.store(name="base_simclr_config", node=SimCLRConfig)


def build_post_epoch_evaluator(
    cfg: EvaluatorConfig,
    device: torch.device,
) -> Callable[[nn.Module], dict[str, float]] | None:
    if cfg.data_dir is None:
        return None

    train_dataset: Dataset[tuple[Tensor, Tensor]] = ImageFolder(
        Path(cfg.data_dir) / cfg.train,
        transform=val_transform,
    )
    validation_dataset: Dataset[tuple[Tensor, Tensor]] = ImageFolder(
        Path(cfg.data_dir) / cfg.validation,
        transform=val_transform,
    )

    train_data_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device == "cuda",
    )
    validation_data_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device == "cuda",
    )

    assert isinstance(train_dataset, Sized)
    assert isinstance(validation_dataset, Sized)
    logger.info("Loaded knn train dataset of %d data points", len(train_dataset))
    logger.info("Loaded knn validation dataset of %d data points", len(validation_dataset))

    def evaluate(model: nn.Module) -> dict[str, float]:
        if isinstance(model, ModelWithHead):
            model = model.model  # Remove head from ModelWithHead.

        evaluator: KNNEvaluator = KNNEvaluator(validation_data_loader, train_data_loader, model)
        accuracy: float = evaluator.evaluate(k=cfg.knn_k)
        return {"accuracy": accuracy}

    return evaluate


@hydra.main(version_base=None, config_name="base_simclr_config")
def main(cfg: SimCLRConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)

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

    evaluator = build_post_epoch_evaluator(cfg.evaluator, device)

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
        evaluator=evaluator,
    )

    # Initialize mlflow and create run context.
    mlflow.set_tracking_uri(Path.cwd() / "runs")
    mlflow.set_experiment(cfg.experiment_tag)

    model_tag = cfg.model_tag or f"{cfg.backbone.model_type.value}_simclr"

    with mlflow.start_run(run_name=model_tag):
        # Log configuration parameters.
        log_hydra_config_to_mlflow(cfg)
        simclr.train(train_loader)

    model.save_backbone(Path(cfg.model_dir) / model_tag)
