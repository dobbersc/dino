from pathlib import Path

import pytest
import torch

from dino.models import HeadType, ModelType, _get_embed_dim, load_model_with_head


@pytest.mark.parametrize(
    (
        "model_type",
        "head_type",
    ),
    [
        (ModelType.VIT_DINO_S, HeadType.LINEAR),
        (ModelType.VIT_T, HeadType.LINEAR),
        (ModelType.VIT_S, HeadType.LINEAR),
        (ModelType.VIT_B, HeadType.LINEAR),
        (ModelType.RESNET_50, HeadType.LINEAR),
        (ModelType.VIT_DINO_S, HeadType.MLP),
        (ModelType.VIT_T, HeadType.MLP),
        (ModelType.VIT_S, HeadType.MLP),
        (ModelType.VIT_B, HeadType.MLP),
        (ModelType.RESNET_50, HeadType.MLP),
    ],
)
def test_load_model_with_head_no_weights(model_type, head_type):
    """Test loading the model with different types without pretrained weights."""
    print("loading model with head: ", model_type, head_type)
    num_classes = 10

    # Load the model without weights
    model_with_head = load_model_with_head(
        model_type=model_type,
        head_type=head_type,
        output_dim=num_classes,
        head_weights=None,
        backbone_weights=None,
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
        (ModelType.VIT_T, HeadType.LINEAR),
        (ModelType.VIT_S, HeadType.LINEAR),
        (ModelType.VIT_B, HeadType.LINEAR),
        (ModelType.RESNET_50, HeadType.LINEAR),
        (ModelType.VIT_DINO_S, HeadType.MLP),
        (ModelType.VIT_T, HeadType.MLP),
        (ModelType.VIT_S, HeadType.MLP),
        (ModelType.VIT_B, HeadType.MLP),
        (ModelType.RESNET_50, HeadType.MLP),
    ],
)
def test_load_model_with_head_with_weights(tmp_model_dir: Path, model_type, head_type):
    """Test loading the model with pretrained weights for backbone and head."""
    print("loading model with head and weights: ", model_type, head_type)
    num_classes = 10
    embed_dim = _get_embed_dim(model_type)

    # Create a dummy model and save weights
    dummy_model_with_head = load_model_with_head(
        model_type=model_type,
        head_type=head_type,
        output_dim=num_classes,
    )

    for param in dummy_model_with_head.parameters():
        param.data.fill_(0.5)  # pseudo training for

    backbone_path = tmp_model_dir / "backbone.pth"
    head_path = tmp_model_dir / "head.pth"

    dummy_model_with_head.save_backbone(backbone_path)
    dummy_model_with_head.save_head(head_path)

    # Load the model with saved weights
    model_with_head = load_model_with_head(
        model_type=model_type,
        head_type=head_type,
        output_dim=num_classes,
        head_weights=head_path,
        backbone_weights=backbone_path,
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
