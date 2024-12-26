import logging
import sys
from dataclasses import dataclass, field

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dino.datasets import DatasetConfig, TransformType, get_dataset
from dino.evaluators import KNNEvaluator, LinearEvaluator
from dino.finetuning import FinetuningMode, finetune
from dino.models import BackboneConfig, HeadConfig, HeadType, load_backbone, load_model_with_head
from dino.utils.random import set_seed

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    batch_size: int = 32

    # KNN specific
    k: int = 5

    # Linear specific
    topk: tuple[int, ...] = (1,)
    num_classes: int | None = None  # tries to infer from dataset
    base_lr: float = 1e-3
    backbone_lr: float = 1e-4
    finetuning_mode: FinetuningMode = FinetuningMode.LINEAR_PROBE
    num_epochs: int = 10
    skip_knn: bool = False
    skip_linear: bool = False


_cs = ConfigStore.instance()
_cs.store(
    name="base_evaluate",
    node=EvaluationConfig,
)


@hydra.main(version_base=None, config_path="../conf", config_name="evaluation_config")
def evaluate(cfg: EvaluationConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    set_seed(42)

    # =========== KNN evaluation ===================
    if not cfg.skip_knn:
        cfg.dataset.train = True
        cfg.dataset.transform = TransformType.KNN
        train_ds = get_dataset(cfg.dataset)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Loaded knn train dataset: len(train_ds)=%d", len(train_ds))

        cfg.dataset.train = False
        eval_ds = get_dataset(cfg.dataset)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Loaded knn eval dataset: len(eval_ds)=%d", len(eval_ds))

        logger.info("Running KNN evaluation")
        knn_model = load_backbone(cfg.backbone)
        evaluator = KNNEvaluator(eval_loader, train_loader, knn_model)
        accuracy = evaluator.evaluate(k=cfg.k)
        logger.info("KNN accuracy: %.2f", accuracy)

    # =========== linear evaluation ================
    if not cfg.skip_linear:
        cfg.dataset.train = True
        cfg.dataset.transform = TransformType.LINEAR_TRAIN
        train_ds = get_dataset(cfg.dataset)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Loaded linear train dataset: len(train_ds)=%d", len(train_ds))

        cfg.dataset.transform = TransformType.LINEAR_VAL
        cfg.dataset.train = False
        eval_ds = get_dataset(cfg.dataset)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Loaded linear eval dataset: len(eval_ds)=%d", len(eval_ds))

        logger.info("Running linear evaluation")
        output_dim = cfg.num_classes if cfg.num_classes is not None else getattr(train_ds, "num_classes", None)
        if output_dim is None:
            logger.error("Could not infer number of classes for linear evaluation")
            sys.exit(1)
        linear_model = load_model_with_head(
            backbone_cfg=cfg.backbone,
            head_cfg=HeadConfig(
                model_type=HeadType.LINEAR,
                output_dim=output_dim,
            ),
        )
        finetune(
            model=linear_model,
            dataloader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            base_lr=cfg.base_lr,
            backbone_lr=cfg.backbone_lr,
            mode=cfg.finetuning_mode,
            num_epochs=cfg.num_epochs,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        evaluator = LinearEvaluator(eval_loader, linear_model)
        accuracies = evaluator.evaluate(topk=cfg.topk)
        logger.info("Linear evaluation accuracies: %s", accuracies)
