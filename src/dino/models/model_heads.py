from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from torch import nn

from dino.models.dino_vision_transformer import vit_small
from dino.utils.torch import load_model, save_model


class ModelType(Enum):
    VIT_DINO_S = "vit-dino-s"
    # VIT = "vit"
    # RESNET = "resnet" # Currently not supported TODO: implement


class HeadType(Enum):
    LINEAR = "linear"
    MLP = "mlp"


@dataclass
class BackboneConfig:
    pretrained_weights: str | None
    model_type: ModelType
    torchhub: tuple[str, str] | None


@dataclass
class HeadConfig:
    model_type: HeadType
    num_classes: int | None


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

    def __init__(self, embed_dim: int, num_classes: int):
        """Initializes the LinearHead module.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_classes (int): The number of output classes.
        """
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the linear head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.head(x)

    @property
    def out_features(self) -> int:
        """Returns the number of output features (i.e., the number of classes).

        Returns:
            int: The number of output features.
        """
        return self.head.out_features

    @property
    def in_features(self) -> int:
        """Returns the number of input features (i.e., the embedding dimension).

        Returns:
            int: The number of input features.
        """
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
                mlp_layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*mlp_layers)
        self.norm = L2NormLayer()

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m: nn.Module):
        """Initializes weights of the linear layers.

        Args:
            m (nn.Module): A module to be initialized if it is a linear layer.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the DINOHead.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.mlp(x)
        x = self.norm(x)
        return self.last_layer(x)


class DINOHeadNormalized(DINOHead):
    """DINO head with weight normalization and frozen scaling parameter."""

    def __init__(self, *args, **kwargs):
        """Initializes the DINOHeadNormalized module.

        Args:
            *args: Positional arguments passed to the base DINOHead class.
            **kwargs: Keyword arguments passed to the base DINOHead class.
        """
        super().__init__(*args, **kwargs)
        self.last_layer.weight_g.requires_grad = False


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
        """Returns the input dimension of the head (embedding dimension).

        Returns:
            int: The input dimension of the head.
        """
        return self.head.in_features

    @property
    def num_classes(self) -> int:
        """Returns the output dimension of the head (number of classes).

        Returns:
            int: The number of classes.
        """
        return self.head.out_features

    def backbone_parameters(self):
        """Returns the parameters of the backbone model.

        Returns:
            Iterator: An iterator over the backbone model parameters.
        """
        return self.model.parameters()

    def head_parameters(self):
        """Returns the parameters of the head.

        Returns:
            Iterator: An iterator over the head parameters.
        """
        return self.head.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the backbone and head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the head.
        """
        x = self.model(x)
        return self.head(x)

    def save_head(self, model_path: str | Path):
        """Saves the head model to a file.

        Args:
            model_name (str): The filename to save the head model. Default is "head.pth".
        """
        save_model(self.head, model_path)

    def save_backbone(self, model_path: str | Path):
        """Saves the backbone model to a file.

        Args:
            model_name (str): The filename to save the backbone model. Default is "backbone.pth".
        """
        save_model(self.model, model_path)

    def save(self, model_path: str | Path):
        """Saves the entire model (backbone and head) to a file.

        Args:
            model_name (str): The filename to save the entire model. Default is "model.pth".
        """
        save_model(self, model_path)


def load_model_with_head(
    model_type: ModelType = ModelType.VIT_DINO_S,
    head_type: HeadType = HeadType.LINEAR,
    num_classes: int = 200,
    head_weights: str | None = None,
    backbone_weights: str | None = None,
    backbone_torchhub: tuple[str, str] | None = None,
) -> ModelWithHead:
    backbone = _load_backbone(
        model_type,
        backbone_weights=backbone_weights,
        backbone_torchhub=backbone_torchhub,
    )
    embed_dim = _get_embed_dim(model_type)
    head = _load_head(head_type, embed_dim, num_classes, head_weights)

    return ModelWithHead(backbone, head)


def _load_backbone(
    model_type: ModelType,
    backbone_weights: str | None = None,
    backbone_torchhub: tuple[str, str] | None = None,
) -> nn.Module:
    # TODO: implement multiple model types
    if backbone_torchhub is not None:
        model = torch.hub.load(*backbone_torchhub)

    else:
        match model_type:
            case ModelType.VIT_DINO_S:
                model = vit_small()
            case _:
                msg = f"Model type {model_type} is not supported"
                raise NotImplementedError(msg)

        if backbone_weights is not None:
            state_dict = load_model(backbone_weights)
            model.load_state_dict(state_dict)

    return model


def _load_head(
    head_type: HeadType,
    embed_dim: int,
    num_classes: int,
    head_weights: str | None = None,
) -> nn.Module:
    match head_type:
        case HeadType.LINEAR:
            head = LinearHead(embed_dim, num_classes)
        case HeadType.MLP:
            head = DINOHead(embed_dim, output_dim=num_classes)
        case _:
            msg = f"Head type {head_type} is not supported"
            raise NotImplementedError(msg)

    if head_weights is not None:
        state_dict = load_model(head_weights)
        head.load_state_dict(state_dict)

    return head


def _get_embed_dim(model_type: ModelType) -> int:
    match model_type:
        case ModelType.VIT_DINO_S:
            return 384
        case _:
            msg = f"Model type {model_type} is not supported"
            raise NotImplementedError(msg)
