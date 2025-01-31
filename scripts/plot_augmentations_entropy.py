from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from scripts import EXPORT_PATH, MLFLOW_DIR
from scripts.utils import load_batch_metrics, plot_error_tube


def plot(label_to_metrics: dict[str, pd.DataFrame]) -> None:
    figure, axes = plt.subplots(2, sharex=True)

    for label, metrics in label_to_metrics.items():
        epoch_metrics = metrics.groupby("epoch")[["student_entropy", "teacher_entropy"]].agg(["mean", "std"])
        epochs = epoch_metrics.index.to_numpy()

        plot_kwargs: dict[str, Any] = {"alpha": 0.75, "label": label}

        for idx, model in enumerate(("teacher", "student")):
            entropy_plot = axes[idx].plot(epochs, epoch_metrics[f"{model}_entropy"]["mean"], **plot_kwargs)
            plot_error_tube(entropy_plot[0], error=epoch_metrics[f"{model}_entropy"]["std"], clip_lower=0)

    axes[0].set_ylim(-0.25, 8.25)
    axes[1].set_ylim(-0.25, 8.25)

    axes[0].set_ylabel("Teacher Entropy")
    axes[1].set_ylabel("Student Entropy")
    axes[1].set_xlabel("Epoch")

    axes[0].legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "augmentations_entropy.pdf")


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "246076156900597482"

    metrics: dict[str, pd.DataFrame] = {
        run_name: load_batch_metrics(experiments_directory / run_id / "metrics")
        for run_name, run_id in (
            (r"Default", "f8e73fdc57ee4428b4caf43def4f1b29"),
            (r"Cropping Only", "edb1b4f8bbd043c6a2be1448690c8b8e"),
        )
    }

    plot(metrics)


if __name__ == "__main__":
    main()
