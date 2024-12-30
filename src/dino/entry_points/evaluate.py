import logging
from collections.abc import Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from dino.datasets import TransformType, get_transform
from dino.evaluators import KNNEvaluator, LinearEvaluator
from dino.finetuning import FinetuningMode, finetune
from dino.models import BackboneConfig, HeadConfig, HeadType, load_backbone, load_model_with_head
from dino.utils.random import set_seed
from dino.utils.torch import detect_device

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    # TODO: This is just a temporary hack without the DatasetConfig to get the dataset's original splits working.
    dataset_dir: str = MISSING
    dataset_train: str = "train"
    dataset_validation: str = "val"

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    batch_size: int = 32
    num_workers: int = 8

    # KNN specific
    skip_knn: bool = False
    k: int = 5

    # Linear specific
    skip_linear: bool = False
    topk: tuple[int, ...] = (1,)
    num_classes: int | None = None  # tries to infer from dataset
    base_lr: float = 1e-3
    backbone_lr: float = 1e-4
    finetuning_mode: FinetuningMode = FinetuningMode.LINEAR_PROBE
    num_epochs: int = 10

    model_dir: str | None = None
    model_tag: str | None = None


_cs = ConfigStore.instance()
_cs.store(
    name="base_evaluate",
    node=EvaluationConfig,
)


# TODO: Refactor this function to also be used in the DINO train mode.
def get_dataloaders(
    cfg: EvaluationConfig,
    transform: TransformType,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    train_dataset: Dataset[tuple[Tensor, Tensor]] = ImageFolder(
        Path(cfg.dataset_dir) / cfg.dataset_train,
        transform=get_transform(transform),
    )
    validation_dataset: Dataset[tuple[Tensor, Tensor]] = ImageFolder(
        Path(cfg.dataset_dir) / cfg.dataset_validation,
        transform=get_transform(transform),
    )

    train_data_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    validation_data_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    assert isinstance(train_dataset, Sized)
    assert isinstance(validation_dataset, Sized)
    logger.info("Loaded train dataset of %d data points", len(train_dataset))
    logger.info("Loaded validation dataset of %d data points", len(validation_dataset))

    return train_data_loader, validation_data_loader


@hydra.main(version_base=None, config_path="../conf", config_name="evaluation_config")
def evaluate(cfg: EvaluationConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = cast(EvaluationConfig, OmegaConf.to_object(cfg))

    # =========== KNN evaluation ===================
    if not cfg.skip_knn:
        set_seed(42)

        logger.info("Running KNN evaluation")
        train_data_loader, validation_data_loader = get_dataloaders(cfg, transform=TransformType.KNN)
        knn_model = load_backbone(cfg.backbone)
        logger.info(knn_model)

        knn_evaluator = KNNEvaluator(validation_data_loader, train_data_loader, knn_model)
        accuracy = knn_evaluator.evaluate(k=cfg.k, device=detect_device())
        logger.info("KNN accuracy: %.2f", accuracy)

    # =========== linear evaluation ================
    if not cfg.skip_linear:
        set_seed(42)

        logger.info("Running linear evaluation")
        train_data_loader, validation_data_loader = get_dataloaders(cfg, transform=TransformType.LINEAR_TRAIN)

        train_dataset: ImageFolder = cast(ImageFolder, train_data_loader.dataset)
        output_dim: int = cfg.num_classes or len(train_dataset.classes)

        linear_model = load_model_with_head(
            backbone_cfg=cfg.backbone,
            head_cfg=HeadConfig(
                model_type=HeadType.LINEAR,
                output_dim=output_dim,
            ),
        )
        logger.info(linear_model)

        finetune(
            model=linear_model,
            dataloader=train_data_loader,
            criterion=nn.CrossEntropyLoss(),
            base_lr=cfg.base_lr,
            backbone_lr=cfg.backbone_lr,
            mode=cfg.finetuning_mode,
            num_epochs=cfg.num_epochs,
            device=detect_device(),
        )

        linear_evaluator = LinearEvaluator(validation_data_loader, linear_model)
        accuracies = linear_evaluator.evaluate(topk=cfg.topk)
        logger.info("Linear evaluation accuracies: %s", accuracies)

        if cfg.model_dir is not None:
            model_name = f"{cfg.model_tag}-backbone" if cfg.model_tag else "finetune-backbone"
            linear_model.save_backbone(Path(cfg.model_dir) / model_name)
