import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import timm
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from dino.datasets import UnlabelledDataset
from dino.models import DINOHead, ModelWithHead
from dino.trainer import DINOTrainer
from dino.utils.random import set_seed
from dino.utils.torch import detect_device, save_model

if TYPE_CHECKING:
    from timm.models import ResNet, VisionTransformer
    from torch import Tensor
    from torch.utils.data import Dataset

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_dir: str = str(Path.cwd() / "models")
    model_tag: str | None = None
    dataset_dir: str = MISSING


_cs = ConfigStore.instance()
_cs.store(
    group="train",
    name="base_train",
    node=TrainingConfig,
)


def train(cfg: TrainingConfig) -> None:
    set_seed(42)

    head_hidden_dim: int = 2048
    head_output_dim: int = 4096
    batch_size: int = 128

    pretrained: bool = False
    model_name: str = "deit_small_patch16_224"

    student: ResNet | VisionTransformer
    teacher: ResNet | VisionTransformer
    if "resnet" in model_name:
        student = timm.create_model(model_name, num_classes=0, pretrained=pretrained)
        teacher = timm.create_model(model_name, num_classes=0, pretrained=pretrained)
    else:
        student = timm.create_model(
            model_name,
            num_classes=0,
            dynamic_img_size=True,
            pretrained=pretrained,
        )
        teacher = timm.create_model(
            model_name,
            num_classes=0,
            dynamic_img_size=True,
            pretrained=pretrained,
        )

    msg = f"{cfg.dataset_dir}; {model_name=}; {pretrained=}"
    logger.info(msg)

    student_with_head: ModelWithHead = ModelWithHead(
        model=student,
        head=DINOHead(
            input_dim=student.num_features,
            output_dim=head_output_dim,
            hidden_dim=head_hidden_dim,
        ),
    )
    teacher_with_head: ModelWithHead = ModelWithHead(
        model=teacher,
        head=DINOHead(
            input_dim=teacher.num_features,
            output_dim=head_output_dim,
            hidden_dim=head_hidden_dim,
        ),
    )

    teacher_with_head.load_state_dict(student_with_head.state_dict())

    dataset: Dataset[Tensor] = UnlabelledDataset(
        dataset=ImageFolder(cfg.dataset_dir, transform=v2.ToImage()),
    )

    trainer: DINOTrainer = DINOTrainer(
        student=student_with_head,
        teacher=teacher_with_head,
        dataset=dataset,
    )
    trainer.train(
        max_epochs=100,
        batch_size=batch_size,
        loss_function_kwargs={"output_size": head_output_dim},
        device=detect_device(),
        optimizer_class=AdamW,
        optimizer_kwargs={
            "lr": 0.0005 * batch_size / 256,
            "weight_decay": 0.04,  # Max: 0.4 TODO: Scheduler this as well
        },
        num_workers=8,
    )

    student_name = f"{cfg.model_tag}_student" if cfg.model_tag else "student"
    save_model(student_with_head, Path(cfg.model_dir) / student_name)

    teacher_name = f"{cfg.model_tag}_teacher" if cfg.model_tag else "teacher"
    save_model(teacher_with_head, Path(cfg.model_dir) / teacher_name)
