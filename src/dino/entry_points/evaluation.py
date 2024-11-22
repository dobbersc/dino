import logging
from dataclasses import dataclass
from enum import Enum

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader

from dino.datasets import DatasetConfig, get_dataset
from dino.evaluators import KNNEvaluator, LinearEvaluator
from dino.models import BackboneConfig, HeadConfig, load_backbone, load_model_with_head

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    KNN = "knn"
    LINEAR = "linear"


@dataclass
class EvaluatorConfig:
    mode: EvaluationType = MISSING
    dataset: DatasetConfig = MISSING
    backbone: BackboneConfig = MISSING
    batch_size: int = MISSING
    # KNN specific
    k: int = MISSING
    # Linear specific
    topk: tuple[int, ...] = (1,)
    head: HeadConfig = MISSING


_cs = ConfigStore.instance()
_cs.store(
    group="evaluate",
    name="base_evaluate",
    node=EvaluatorConfig,
)


def run_knn(cfg: EvaluatorConfig) -> None:
    logger.info("Running KNN evaluation")

    model = load_backbone(cfg.backbone.model_type, cfg.backbone.pretrained_weights)
    msg = f"Loaded model with backbone {cfg.backbone.model_type}"
    logger.info(msg)

    cfg.dataset.train = True
    train_ds = get_dataset(cfg.dataset)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    msg = f"Loaded train dataset: len(train_ds)={len(train_ds)}"
    logger.info(msg)

    cfg.dataset.train = False
    eval_ds = get_dataset(cfg.dataset)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    msg = f"Loaded eval dataset: len(eval_ds)={len(eval_ds)}"
    logger.info(msg)

    evaluator = KNNEvaluator(eval_loader, train_loader, model)
    accuracy = evaluator.evaluate(k=cfg.k)
    msg = f"KNN accuracy: {accuracy:.2f}"
    logger.info(msg)


def run_linear(cfg: EvaluatorConfig) -> None:
    logger.info("Running linear evaluation")
    model = load_model_with_head(
        model_type=cfg.backbone.model_type,
        head_type=cfg.head.model_type,
        output_dim=cfg.head.output_dim,
        hidden_dim=cfg.head.hidden_dim,
        backbone_weights=cfg.backbone.pretrained_weights,
        head_weights=cfg.head.pretrained_weights,
    )
    msg = f"Loaded model with backbone {cfg.backbone.model_type} and head {cfg.head.model_type}"
    logger.info(msg)

    cfg.dataset.train = False
    eval_ds = get_dataset(cfg.dataset)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    msg = f"Loaded eval dataset: len(eval_ds)={len(eval_ds)}"
    logger.info(msg)

    evaluator = LinearEvaluator(eval_loader, model)
    accuracies = evaluator.evaluate(topk=cfg.topk)
    msg = f"Linear evaluation accuracies: {accuracies}"
    logger.info(msg)


def run_evaluation(cfg: EvaluatorConfig) -> None:
    match cfg.mode:
        case EvaluationType.KNN:
            run_knn(cfg)
        case EvaluationType.LINEAR:
            run_linear(cfg)
        case _:
            msg = f"Evaluation type {cfg.type} is not supported"
            raise NotImplementedError(msg)
