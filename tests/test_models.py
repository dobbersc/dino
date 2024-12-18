import logging
from pathlib import Path

import pytest
import torch

from dino.models import (
    BackboneConfig,
    HeadConfig,
    HeadType,
    ModelType,
    _get_embed_dim,
    load_model_with_head,
)


@pytest.mark.parametrize(
    (
        "model_type",
        "head_type",
    ),
    [
        (ModelType.VIT_DINO_S, HeadType.LINEAR),
        (ModelType.DEIT_S, HeadType.LINEAR),
        (ModelType.RESNET_50, HeadType.LINEAR),
        (ModelType.VIT_DINO_S, HeadType.DINO_HEAD),
        (ModelType.DEIT_S, HeadType.DINO_HEAD),
        (ModelType.RESNET_50, HeadType.DINO_HEAD),
    ],
)
def test_load_model_with_head_no_weights(model_type, head_type):
    """Test loading the model with different types without pretrained weights."""
    print("loading model with head: ", model_type, head_type)
    num_classes = 10

    # Load the model without weights
    model_with_head = load_model_with_head(
        backbone_cfg=BackboneConfig(model_type=model_type),
        head_cfg=HeadConfig(
            model_type=head_type,
            output_dim=num_classes,
        ),
    )

    assert model_with_head is not None, "Model should not be None"
    assert model_with_head.output_dim == num_classes, "Number of classes should match"
    assert model_with_head.embed_dim == _get_embed_dim(
        model_type,
    ), "Embedding dimension should match"


@pytest.mark.parametrize(
    (
        "model_type",
        "head_type",
    ),
    [
        (ModelType.VIT_DINO_S, HeadType.LINEAR),
        (ModelType.DEIT_S, HeadType.LINEAR),
        (ModelType.RESNET_50, HeadType.LINEAR),
        (ModelType.VIT_DINO_S, HeadType.DINO_HEAD),
        (ModelType.DEIT_S, HeadType.DINO_HEAD),
        (ModelType.RESNET_50, HeadType.DINO_HEAD),
    ],
)
def test_load_model_with_head_with_weights(tmp_model_dir: Path, model_type, head_type, caplog):
    """Test loading the model with pretrained weights for backbone and head."""
    print("loading model with head and weights: ", model_type, head_type)
    caplog.set_level(logging.INFO)
    num_classes = 10
    embed_dim = _get_embed_dim(model_type)

    # Create a dummy model and save weights
    dummy_model_with_head = load_model_with_head(
        backbone_cfg=BackboneConfig(model_type=model_type),
        head_cfg=HeadConfig(
            model_type=head_type,
            output_dim=num_classes,
        ),
    )

    for param in dummy_model_with_head.parameters():
        param.data.fill_(0.5)  # pseudo training for

    assert tmp_model_dir.exists(), f"Temporary directory not created: {tmp_model_dir}"

    backbone_path = tmp_model_dir / "backbone.pt"
    head_path = tmp_model_dir / "head.pt"

    dummy_model_with_head.save_backbone(backbone_path)
    dummy_model_with_head.save_head(head_path)

    # Verify files are saved
    assert backbone_path.exists(), f"Backbone file not saved: {backbone_path}"
    assert head_path.exists(), f"Head file not saved: {head_path}"

    # Load the model with saved weights
    model_with_head = load_model_with_head(
        backbone_cfg=BackboneConfig(
            model_type=model_type,
            weights=backbone_path,
        ),
        head_cfg=HeadConfig(
            model_type=head_type,
            output_dim=num_classes,
            weights=head_path,
        ),
    )

    assert model_with_head is not None, "Model should not be None"
    assert model_with_head.output_dim == num_classes, "Number of classes should match"
    assert model_with_head.embed_dim == embed_dim, "Embedding dimension should match"

    # Check if weights were loaded correctly
    for param1, param2 in zip(
        dummy_model_with_head.parameters(),
        model_with_head.parameters(),
        strict=False,
    ):
        assert torch.equal(param1, param2), "Model parameters should be identical"
