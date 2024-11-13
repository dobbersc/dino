# my_package/__main__.py

import argparse

from finetuning import run_finetuning
from hydra import compose, initialize


def train(args):
    print("Training model...")
    # TODO: Add your training logic here


def add_train_subparser(subparsers):
    # Subcommand for training
    train_parser = subparsers.add_parser("train", help="Train the model.")
    # train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    # train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")


def evaluate(args):
    print("Evaluating model...")
    # TODO: Add your evaluation logic here


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
    add_evaluate_subparser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command

    match args.command:
        case "train":
            train(args)
        case "evaluate":
            evaluate(args)
        case "finetune":
            with initialize(config_path="conf/finetune"):
                cfg = compose(config_name="config")
                run_finetuning(cfg)
        case _:
            print("Invalid command.")


if __name__ == "__main__":
    entry_point()
