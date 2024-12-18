import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from dino.datasets import DatasetConfig, UnlabelledDataset
from dino.models import BackboneConfig, HeadConfig, load_model_with_head
from dino.trainer import DINOTrainer
from dino.utils.random import set_seed
from dino.utils.schedulers import (
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    Scheduler,
    SequentialScheduler,
)
from dino.utils.torch import detect_device

logger = logging.getLogger(__name__)


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

    teacher_momentum_initial: float = 0.996
    teacher_momentum_final: float | None = (
        1.0  # teacher momentum scheduler final value (optional) -> Constant or Cosine Scheduler
    )

    teacher_temp: float = 0.04  # teacher temperature, constant value
    teacher_temp_warmup: float = (
        0.04  # teacher temperature initial warmup value, linearly increasing
    )
    teacher_temp_warmup_epochs: int = 0  # teacher temperature warmup epochs

    center_momentum: float = 0.9  # center momentum constant value


_cs = ConfigStore.instance()
_cs.store(
    name="base_train",
    node=TrainingConfig,
)


@hydra.main(version_base=None, config_path="../conf", config_name="train_config")
def train(cfg: TrainingConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    set_seed(42)

    student_with_head = load_model_with_head(cfg.backbone, cfg.head)
    teacher_with_head = load_model_with_head(cfg.backbone, cfg.head)
    msg = f"{cfg.dataset.data_dir}; {cfg.backbone.model_type=}; {cfg.backbone.pretrained=}"
    logger.info(msg)

    teacher_with_head.load_state_dict(student_with_head.state_dict())

    # for more flexible dataset configuration: dataset = dino.datasets.get_dataset(cfg.dataset)
    dataset: Dataset[Tensor] = UnlabelledDataset(
        dataset=ImageFolder(cfg.dataset.data_dir, transform=v2.ToImage()),
    )

    trainer: DINOTrainer = DINOTrainer(
        student=student_with_head,
        teacher=teacher_with_head,
        dataset=dataset,
    )

    # the step counter is increased after each batch for teacher_momentum and loss_function
    # The dino loss updates the the teacher's temperature and center momentum for each step.
    # calculate n_batches per epoch for the internal view_dataset:
    steps_per_epoch = len(trainer.view_dataset) // cfg.batch_size

    max_steps: int = steps_per_epoch * cfg.max_epochs
    teacher_momentum: Scheduler[float] = (
        ConstantScheduler(constant=cfg.teacher_momentum_initial)
        if cfg.teacher_momentum_final is None
        else CosineScheduler(
            max_steps=max_steps,
            initial=cfg.teacher_momentum_initial,
            final=cfg.teacher_momentum_final,
        )
    )

    msg = f"Using teacher momentum scheduler: {type(teacher_momentum).__name__}"
    logger.info(msg)

    teacher_temp = ConstantScheduler(cfg.teacher_temp)
    if cfg.teacher_temp_warmup_epochs > 0:
        teacher_temp = SequentialScheduler(
            [
                # insert linearly increasing warmup scheduler
                LinearScheduler(
                    cfg.teacher_temp_warmup,
                    max_steps=steps_per_epoch * cfg.teacher_temp_warmup_epochs,
                ),
                teacher_temp,
            ],
        )
    msg = f"Using teacher temperature scheduler: {type(teacher_temp).__name__}"
    logger.info(msg)

    trainer.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        loss_function_kwargs={
            "output_size": cfg.head.output_dim,
            "teacher_temperature": teacher_temp,
            "center_momentum": ConstantScheduler(cfg.center_momentum),
        },
        device=detect_device(),
        optimizer_class=AdamW,
        optimizer_kwargs={
            "lr": 0.0005 * cfg.batch_size / 256,
            "weight_decay": 0.04,  # Max: 0.4 TODO: Scheduler this as well
        },
        teacher_momentum=teacher_momentum,
        num_workers=8,
    )

    student_name = f"{cfg.model_tag}_student" if cfg.model_tag else "student"
    student_with_head.save_backbone(Path(cfg.model_dir) / student_name)

    teacher_name = f"{cfg.model_tag}_teacher" if cfg.model_tag else "teacher"
    teacher_with_head.save(Path(cfg.model_dir) / teacher_name)
