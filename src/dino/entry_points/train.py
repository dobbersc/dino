import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from dino.datasets import DatasetConfig, UnlabelledDataset
from dino.models import BackboneConfig, HeadConfig, ModelWithHead, load_model_with_head
from dino.trainer import DINOTrainer
from dino.utils.logging import log_hydra_config_to_mlflow
from dino.utils.random import set_seed
from dino.utils.schedulers import (
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    Scheduler,
    SequentialScheduler,
)
from dino.utils.torch import detect_device

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_dir: str = str(Path.cwd() / "models")
    model_tag: str | None = None

    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)

    batch_size: int = 128
    max_epochs: int = 100

    # Teacher momentum scheduler values:
    # If the final value is set to None, a constant scheduler with the initial value will be used.
    # Otherwise, a cosine scheduler from the initial to the final value will be used.
    teacher_momentum_initial: float = 0.996
    teacher_momentum_final: float | None = 1.0

    # Teacher temperature linear warmup scheduler:
    # If the final value is set to None, a constant scheduler with the initial value will be used.
    # Otherwise, a constant scheduler with a linear warmup from the initial to the final value will be used.
    teacher_temperature_initial: float = 0.04
    teacher_temperature_final: float | None = 0.07
    teacher_temperature_warmup_epochs: int = 30

    center_momentum: float = 0.9  # Constant momentum for the teacher's logits center.

    num_workers: int = 8


_cs = ConfigStore.instance()
_cs.store(
    name="base_train",
    node=TrainingConfig,
)


@hydra.main(version_base=None, config_path="../conf", config_name="train_config")
def train(cfg: TrainingConfig) -> None:
    set_seed(42)

    logger.info(OmegaConf.to_yaml(cfg))
    cfg = cast(TrainingConfig, OmegaConf.to_object(cfg))

    logger.info("%s; %s, %s", f"{cfg.dataset.data_dir=}", f"{cfg.backbone.model_type=}", f"{cfg.backbone.pretrained=}")

    student_with_head: ModelWithHead = load_model_with_head(cfg.backbone, cfg.head)
    teacher_with_head: ModelWithHead = load_model_with_head(cfg.backbone, cfg.head)
    teacher_with_head.load_state_dict(student_with_head.state_dict())

    # for more flexible dataset configuration: dataset = dino.datasets.get_dataset(cfg.dataset)
    dataset: Dataset[Tensor] = UnlabelledDataset(
        dataset=ImageFolder(cfg.dataset.data_dir, transform=v2.ToImage()),
    )

    trainer: DINOTrainer = DINOTrainer(
        student=student_with_head,
        teacher=teacher_with_head,
        view_dataset=dataset,
    )

    # The step counter is increased after each batch for the dino trainer's schedulers.
    # Since the trainer's data loader drops the last data points such that the effective dataset is
    # divisible by the batch size, we can easily calculate the maximum number of steps.
    steps_per_epoch: int = len(trainer.view_dataset) // cfg.batch_size
    max_steps: int = steps_per_epoch * cfg.max_epochs

    teacher_momentum: Scheduler[float] = (
        ConstantScheduler(max_steps=max_steps, constant=cfg.teacher_momentum_initial)
        if cfg.teacher_momentum_final is None
        else CosineScheduler(
            max_steps=max_steps,
            initial=cfg.teacher_momentum_initial,
            final=cfg.teacher_momentum_final,
        )
    )
    logger.info("Using teacher momentum scheduler: %s", teacher_momentum)

    if cfg.teacher_temperature_warmup_epochs > cfg.max_epochs:
        msg: str = (
            f"The number of warmup epochs for the teacher temperature ({cfg.teacher_temperature_warmup_epochs}) "
            f"must not be greater than the number of maximum epochs ({cfg.max_epochs})."
        )
        raise RuntimeError(msg)

    teacher_temperature: Scheduler[float]
    if cfg.teacher_temperature_final is None or cfg.teacher_temperature_warmup_epochs <= 0:
        teacher_temperature = ConstantScheduler(max_steps=max_steps, constant=cfg.teacher_temperature_initial)
    else:
        milestone: int = cfg.teacher_temperature_warmup_epochs * steps_per_epoch
        teacher_temperature = SequentialScheduler(
            schedulers=[
                LinearScheduler(
                    max_steps=milestone,
                    initial=cfg.teacher_temperature_initial,
                    final=cfg.teacher_temperature_final,
                ),
                ConstantScheduler(max_steps=max_steps - milestone, constant=cfg.teacher_temperature_final),
            ],
            milestones=[milestone],
        )
    logger.info("Using teacher temperature scheduler: %s", teacher_temperature)

    # Initialize mlflow and create run context.
    mlflow.set_tracking_uri(Path.cwd() / "runs")
    mlflow.set_experiment("model_training")

    with mlflow.start_run(run_name=f"dino training {cfg.model_tag}"):
        # Log configuration parameters.
        log_hydra_config_to_mlflow(cfg)

        trainer.train(
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            loss_function_kwargs={
                "output_size": cfg.head.output_dim,
                "teacher_temperature": teacher_temperature,
                "center_momentum": ConstantScheduler(max_steps=max_steps, constant=cfg.center_momentum),
            },
            device=detect_device(),
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 0.0005 * cfg.batch_size / 256},
            teacher_momentum=teacher_momentum,
            num_workers=cfg.num_workers,
        )

    student_name = f"{cfg.model_tag}_student" if cfg.model_tag else "student"
    student_with_head.save_backbone(Path(cfg.model_dir) / student_name)

    teacher_name = f"{cfg.model_tag}_teacher" if cfg.model_tag else "teacher"
    teacher_with_head.save(Path(cfg.model_dir) / teacher_name)


if __name__ == "__main__":
    train()
