from enum import Enum

import torch
from torch import nn

from dino import utils
from dino.models.dino_vision_transformer import vit_small


class LinearHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x)

    @property
    def out_features(self):
        return self.head.out_features

    @property
    def in_features(self):
        return self.head.in_features

    def parameters(self):
        return self.head.parameters()


class ModelWithHead(nn.Module):
    def __init__(self, model: nn.Module, head: nn.Module):
        super().__init__()
        self.model = model
        self.head = head

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self):
        return self.head.in_features

    @property
    def num_classes(self):
        return self.head.out_features

    def backbone_paramters(self):
        return self.model.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, x):
        x = self.model(x)
        return self.head(x)

    def save_head(self, model_name="head.pth"):
        utils.save_model(self.head, model_name)

    def save_backbone(self, model_name="backbone.pth"):
        utils.save_model(self.model, model_name)

    def save(self, model_name="model.pth"):
        utils.save_model(self, model_name)


class ModelType(Enum):
    VIT_DINO_S = "vit-dino-s"
    VIT = "vit"
    RESNET = "resnet"


class HeadType(Enum):
    LINEAR = "linear"
    MLP = "mlp"
    KNN = "knn"


def load_model_with_head(
    model_type: ModelType = ModelType.VIT_DINO_S,
    num_classes: int = 200,
    head_weights: str | None = None,
    backbone_weights: str | None = None,
    backbone_torchhub: tuple[str, str] | None = None,
) -> ModelWithHead:
    backbone = _load_backbone(
        model_type, backbone_weights=backbone_weights, backbone_torchhub=backbone_torchhub
    )
    embed_dim = _get_embed_dim(model_type)
    head = _load_head(HeadType.LINEAR, embed_dim, num_classes, head_weights)

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
            state_dict = utils.load_model(backbone_weights)
            model.load_state_dict(state_dict)

    return model


def _load_head(
    head_type: HeadType, embed_dim: int, num_classes: int, head_weights: str | None = None
) -> nn.Module:
    if head_type == HeadType.LINEAR:
        head = LinearHead(embed_dim, num_classes)
    else:
        msg = f"Head type {head_type} is not supported"
        raise NotImplementedError(msg)

    if head_weights is not None:
        state_dict = utils.load_model(head_weights)
        head.load_state_dict(state_dict)

    return head


def _get_embed_dim(model_type: ModelType) -> int:
    match model_type:
        case ModelType.VIT_DINO_S:
            return 384
        case _:
            msg = f"Model type {model_type} is not supported"
            raise NotImplementedError(msg)
