from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import timm
import torch
from hydra.core.config_store import ConfigStore
from torch import nn

from dino.utils.torch import load_model, save_model


class ModelType(Enum):
    VIT_DINO_S = "vit-dino-s"
    DEIT_S = "deit-s"
    RESNET_50 = "resnet-50"


class HeadType(Enum):
    LINEAR = "linear"
    DINO_HEAD = "dino_head"
    SIMCLR_HEAD = "simclr_head"


@dataclass
class BackboneConfig:
    model_type: ModelType = ModelType.DEIT_S
    weights: str | None = None
    pretrained: bool = False


@dataclass
class HeadConfig:
    model_type: HeadType = HeadType.DINO_HEAD
    output_dim: int = 4096
    hidden_dim: int = 2048
    weights: str | None = None


_cs = ConfigStore.instance()
_cs.store(
    group="head",
    name="base_head",
    node=HeadConfig,
)
_cs.store(
    group="backbone",
    name="base_backbone",
    node=BackboneConfig,
)


class LinearHead(nn.Module):
    """A simple linear classification head (embeddings to number of classes)."""

    def __init__(self, embed_dim: int, output_dim: int):
        """Initializes the LinearHead module.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            output_dim (int): The number of output classes.
        """
        super().__init__()
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the linear head."""
        return self.head(x)

    @property
    def out_features(self) -> int:
        """Returns the number of output features (i.e., the number of classes)."""
        return self.head.out_features

    @property
    def in_features(self) -> int:
        """Returns the number of input features (i.e., the embedding dimension)."""
        return self.head.in_features


class L2NormLayer(nn.Module):
    """A layer that performs L2 normalization on the input tensor along the specified dimension."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)


class DINOHead(nn.Module):
    """MLP-based head for DINO with optional L2 normalization and weight-normalized output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ):
        """Initializes the DINOHead module.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            n_layers (int): Number of layers in the MLP. Default is 3.
        """
        super().__init__()
        n_layers = max(1, n_layers)

        mlp_layers = []
        for i in range(n_layers):
            mlp_layers.append(
                nn.Linear(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim if i < n_layers - 1 else bottleneck_dim,
                ),
            )
            if i < n_layers - 1:
                mlp_layers.append(nn.GELU())

        self.mlp = nn.Sequential(*mlp_layers)

        self.apply(self._init_weights)
        self.norm = L2NormLayer()

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m: nn.Module):
        """Initializes weights of the linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the DINOHead."""
        x = self.mlp(x)
        x = self.norm(x)
        return self.last_layer(x)

    @property
    def out_features(self) -> int:
        """Returns the number of output features (i.e., the number of classes)."""
        return self.last_layer.out_features

    @property
    def in_features(self) -> int:
        """Returns the number of input features (i.e., the embedding dimension)."""
        return self.mlp[0].in_features


class DINOHeadNormalized(DINOHead):
    """DINO head with weight normalization and frozen scaling parameter."""

    def __init__(self, *args, **kwargs):
        """Initializes the DINOHeadNormalized module."""
        super().__init__(*args, **kwargs)
        self.last_layer.weight_g.requires_grad = False

class SimCLRHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        """A projection head for SimCLR with a two-layer MLP."""
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.projection(x)

class ModelWithHead(nn.Module):
    """Combines a backbone model with a classification head."""

    def __init__(self, model: nn.Module, head: nn.Module):
        """Initializes the ModelWithHead.

        Args:
            model (nn.Module): The backbone model.
            head (nn.Module): The classification head.
        """
        super().__init__()
        self.model = model
        self.head = head

    def freeze_backbone(self):
        """Freezes the backbone model, preventing its parameters from being updated during training."""
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self) -> int:
        """Returns the input dimension of the head (embedding dimension)."""
        return self.head.in_features

    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the head (number of classes)."""
        return self.head.out_features

    def backbone_parameters(self):
        """Returns the parameters of the backbone model."""
        return self.model.parameters()

    def head_parameters(self):
        """Returns the parameters of the head."""
        return self.head.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the backbone and head."""
        x = self.model(x)
        return self.head(x)

    def save_head(self, model_path: str | Path):
        """Saves the head model to a file."""
        save_model(self.head, model_path)

    def save_backbone(self, model_path: str | Path):
        """Saves the backbone model to a file."""
        save_model(self.model, model_path)

    def save(self, model_path: str | Path):
        """Saves the entire model (backbone and head) to a file."""
        save_model(self, model_path)


def load_model_with_head(
    backbone_cfg: BackboneConfig,
    head_cfg: HeadConfig,
) -> ModelWithHead:
    backbone = load_backbone(backbone_cfg)
    embed_dim = (
        backbone.num_features
        if hasattr(backbone, "num_features")
        else _get_embed_dim(backbone_cfg.model_type)
    )
    head = _load_head(
        head_cfg.model_type,
        embed_dim,
        head_cfg.output_dim,
        head_cfg.hidden_dim,
        head_cfg.weights,
    )

    return ModelWithHead(backbone, head)


def load_backbone(cfg: BackboneConfig) -> nn.Module:
    match cfg.model_type:
        case ModelType.VIT_DINO_S:
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        case ModelType.DEIT_S:
            model = timm.create_model(
                "deit_small_patch16_224",
                num_classes=0,
                dynamic_img_size=True,
                pretrained=cfg.pretrained,
            )
        case ModelType.RESNET_50:
            model = timm.create_model(
                "resnet50",
                num_classes=0,
                pretrained=cfg.pretrained,
            )
        case _:
            msg = f"Model type {cfg.model_type} is not supported"
            raise NotImplementedError(msg)

    if cfg.weights is not None:
        state_dict = load_model(cfg.weights)
        model.load_state_dict(state_dict)

    return model


def _load_head(
    head_type: HeadType,
    embed_dim: int,
    output_dim: int,
    hidden_dim: int = 2048,
    weights: str | None = None,
) -> nn.Module:
    match head_type:
        case HeadType.LINEAR:
            head = LinearHead(embed_dim, output_dim)
        case HeadType.DINO_HEAD:
            head = DINOHead(embed_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        case HeadType.SIMCLR_HEAD:
            head = SimCLRHead(embed_dim, hidden_dim, output_dim)
        case _:
            msg = f"Head type {head_type} is not supported"
            raise NotImplementedError(msg)

    if weights is not None:
        state_dict = load_model(weights)
        head.load_state_dict(state_dict)

    return head


def _get_embed_dim(model_type: ModelType) -> int:
    match model_type:
        case ModelType.VIT_DINO_S:
            return 384
        case ModelType.DEIT_S:
            return 384
        case ModelType.RESNET_50:
            return 2048
        case _:
            msg = f"Model type {model_type} is not supported"
            raise NotImplementedError(msg)
