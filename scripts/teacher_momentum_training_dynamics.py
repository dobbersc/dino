from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from scripts import EXPORT_PATH, MLFLOW_DIR


def plot(label_to_metrics: dict[str, pd.DataFrame]) -> None:
    figure, axes = plt.subplots(2, sharex=True)

    for index, (label, metrics) in enumerate(label_to_metrics.items()):
        plot_kwargs: dict[str, Any] = {"alpha": 0.75, "linestyle": "--" if index == 0 else "-", "label": label}
        epochs = metrics.index.to_numpy()

        kl_divergence_plot = axes[0].plot(epochs, metrics["kl_divergence"], **plot_kwargs)
        axes[0].fill_between(
            epochs,
            (metrics["kl_divergence"] - metrics["kl_divergence_std"]).clip(lower=0),
            metrics["kl_divergence"] + metrics["kl_divergence_std"],
            color=kl_divergence_plot[0].get_color(),
            alpha=0.25,
        )

        axes[1].plot(epochs, metrics["accuracy"], **plot_kwargs)

    axes[0].set_ylabel("KL Divergence")
    axes[1].set_ylabel("Student kNN Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[0].legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "teacher_momentum_training_dynamics.pdf")


def load_metrics(metrics_directory: Path) -> pd.DataFrame:
    batch_kl_divergences: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_batch_kl_divergence",
        sep=" ",
        usecols=(1, 2),
        names=("kl_divergence", "batch"),
    )
    epochs: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_batch_epoch",
        sep=" ",
        usecols=(1, 2),
        names=("epoch", "batch"),
    )
    batch_kl_divergences = batch_kl_divergences.merge(epochs, on="batch")

    epoch_kl_divergences: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_epoch_kl_divergence",
        sep=" ",
        usecols=(1, 2),
        names=("kl_divergence", "epoch"),
        index_col="epoch",
    )
    epoch_kl_divergences = epoch_kl_divergences.assign(
        kl_divergence_std=batch_kl_divergences.groupby("epoch")["kl_divergence"].std(),
    )

    epoch_student_accuracies: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_epoch_student_accuracy",
        sep=" ",
        usecols=(1, 2),
        names=("accuracy", "epoch"),
        index_col="epoch",
    )

    metrics: pd.DataFrame = epoch_kl_divergences.merge(epoch_student_accuracies, left_index=True, right_index=True)

    return metrics


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "465212383374859638"

    metrics: dict[str, pd.DataFrame] = {
        run_name: load_metrics(experiments_directory / run_id / "metrics")
        for run_name, run_id in (
            (r"$\lambda=0.9$", "50561b4eff294676ad72c02f532e083c"),
            (r"$\lambda=0.95$", "d979db49e56b4264be657ddd6f24e818"),
            (r"$\lambda=0.99$", "b74cecda5efb4fd9b114f12c1056e73b"),
            (r"$\lambda=0.999$", "4d8e365b6f0a43eea9d0ee9665ac00a6"),
        )
    }

    plot(metrics)


if __name__ == "__main__":
    main()
