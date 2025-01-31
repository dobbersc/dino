import functools
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from scripts import MLFLOW_DIR, EXPORT_PATH


# tODO: Add dino

def plot(label_to_metrics: dict[str, pd.DataFrame]) -> None:
    figure, ax = plt.subplots()

    max_epochs: int = 100

    for label, metrics in label_to_metrics.items():
        ax.plot(
            metrics.index.to_numpy()[:max_epochs],
            metrics["top1_train_accuracy"][:max_epochs],
            linestyle="--",
            alpha=0.9,
            label=f"Top-1 Train Accuracy ({label})",
        )

    for label, metrics in label_to_metrics.items():
        ax.plot(
            metrics.index.to_numpy()[:max_epochs],
            metrics["accuracy"][:max_epochs],
            alpha=0.9,
            label=f"kNN Accuracy ({label})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "sim_clr_accuracy.pdf")


def load_metrics(metrics_directory: Path) -> pd.DataFrame:
    metrics: list[pd.DataFrame] = [
        pd.read_csv(
            metrics_directory / f"train_epoch_{metric_name}",
            sep=" ",
            usecols=(1, 2),
            index_col="epoch",
            names=(metric_name, "epoch"),
        )
        for metric_name in ("top1_train_accuracy", "accuracy")
    ]
    return functools.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), metrics)


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "451482724824650695"

    metrics: dict[str, pd.DataFrame] = {
        run_name: load_metrics(experiments_directory / run_id / "metrics")
        for run_name, run_id in (
            (r"DeiT-S", "5d135f66502a46848dda3adc679606b7"),
            (r"ResNet50", "b5aaf71ade1a41aa85a649ee84e98177"),
        )
    }

    plot(metrics)


if __name__ == "__main__":
    main()
