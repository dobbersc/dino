# my_package/__main__.py

import argparse
import ast
import random

import torch.nn as nn
from torch.utils.data import DataLoader

from dino import datasets, utils
from dino.finetuning import FinetuningMode, fine_tune
from dino.models.model_heads import ModelType, load_model_with_head


def train(args):
    print("Training model...")
    # Add your training logic here


def add_train_subparser(subparsers):
    # Subcommand for training
    train_parser = subparsers.add_parser("train", help="Train the model.")
    # train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    # train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")


def finetune(args):
    print("Fine-tuning model...")
    # Add your fine-tuning logic here
    seed = 42
    random.seed(seed)
    if args.dataset == "imagenet":
        dataset = datasets.ImageNetDirectoryDataset(
            utils.IMAGENET_TINY_DIR,
            transform=datasets.transform,
            path_wnids=utils.IMAGENET_TINY_WORDS,
        )
    elif args.dataset == "imagenet-tiny":
        dataset = datasets.ImageNetDirectoryDataset(
            utils.IMAGENET_TINY_DIR,
            transform=datasets.transform,
            path_wnids=utils.IMAGENET_TINY_WORDS,
            num_sample_classes=5,
        )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    NUM_CLASSES = len(dataset.class_idx_to_wnid)
    NUM_SAMPLES = len(dataset)
    mode = FinetuningMode(args.mode)
    model_type = ModelType(args.model_type)

    if args.backbone_torchhub is not None:
        backbone_torchhub = ast.literal_eval(args.backbone_torchhub)
        if not (
            isinstance(backbone_torchhub, tuple)
            and len(backbone_torchhub) == 2
            and all(isinstance(element, str) for element in backbone_torchhub)
        ):
            msg = "backbone_torchhub must be a tuple with exactly two string elements."
            raise ValueError(msg)

        model = load_model_with_head(
            model_type=model_type, backbone_torchhub=backbone_torchhub, num_classes=NUM_CLASSES
        )
    else:
        raise NotImplementedError("Only TorchHub models are supported for now.")
        # TODO: Implement loading models from local files

    print(
        f"Fine-tuning model...\n\tModel: {model_type}\n\tMode: {mode}\n"
        f"\tDataset: {args.dataset}\n\tNum Epochs: {args.num_epochs}\n"
        f"\tBatch Size: {args.batch_size}\n\tnum_samples: {NUM_SAMPLES}"
    )
    criterion = nn.CrossEntropyLoss()

    # TODO: implement checkpointing
    fine_tune(model, dataloader, criterion, mode=mode, num_epochs=args.num_epochs)
    model.save_head(args.model_name)
    print(f"Model saved to {utils.MODEL_DIR}/{args.model_name}")


def add_finetune_subparser(subparsers):
    # Subcommand for fine-tuning
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the model.")
    finetune_parser.add_argument(
        "--mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "full_finetune"],
        help="Mode for fine-tuning.",
    )
    finetune_parser.add_argument(
        "--model_type",
        type=str,
        default="vit-dino-s",
        choices=["vit-dino-s", "vit-dino-b", "resnet50"],
        help="Path to the model to fine-tune.",
    )
    finetune_parser.add_argument(
        "--backbone_torchhub",
        type=str,
        default="('facebookresearch/dino:main', 'dino_vits8')",
        help="Load backbone from TorchHub.",
    )
    finetune_parser.add_argument(
        "--dataset",
        default="imagenet-tiny",
        choices=["imagenet-tiny", "imagenet", "cifar"],
        help="Dataset to use.",
    )
    finetune_parser.add_argument(
        "--model_name", type=str, default="model.pth", help="Name of the model file."
    )
    finetune_parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train."
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )


def evaluate(args):
    print("Evaluating model...")
    # Add your evaluation logic here


def add_evaluate_subparser(subparsers):
    # Subcommand for evaluation
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model.")
    evaluate_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model to evaluate."
    )


def entry_point():
    parser = argparse.ArgumentParser(description="Model management.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_subparser(subparsers)
    add_finetune_subparser(subparsers)
    add_evaluate_subparser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if args.command == "train":
        train(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    entry_point()
