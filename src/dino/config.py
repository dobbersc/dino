from dataclasses import dataclass
from pathlib import Path

from dino.datasets import DatasetType, TransformType
from dino.finetuning import FinetuningMode
from dino.models.model_heads import HeadType, ModelType


@dataclass
class DatasetConfig:
    type_: DatasetType
    transform: TransformType
    data_dir: str | Path = "/input-data"


class ImageNetConfig(DatasetConfig):
    num_sample_classes: int | None
    path_wnids: str | Path | None


@dataclass
class BackboneConfig:
    model_type: ModelType = ModelType.VIT_DINO_S
    torchhub: tuple[str, str] | None = ("facebookresearch/dino:main", "dino_vits8")
    pretrained_weights: str | None


@dataclass
class HeadConfig:
    model_type: HeadType
    num_classes: int | None


@dataclass
class FinetuningConfig:
    base_lr: float = 1e-3
    backbone_lr: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 32
    mode: FinetuningMode = FinetuningMode.LINEAR_PROBE
    dataset: DatasetConfig
    backbone: BackboneConfig
    head: HeadConfig
    model_dir: str | Path
    model_tag: str
