from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from scripts import MLFLOW_DIR, EXPORT_PATH
from scripts.utils import load_epoch_metrics


def plot(label_to_metrics: dict[str, pd.DataFrame]) -> None:
    figure, ax = plt.subplots()

    for label, metrics in label_to_metrics.items():
        ax.plot(metrics.index.to_numpy(), metrics["student_accuracy"], alpha=0.9, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Student kNN Accuracy")

    ax.legend(loc="lower right")

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "number_local_views_accuracy.pdf")


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "246076156900597482"

    metrics: dict[str, pd.DataFrame] = {
        run_name: load_epoch_metrics(experiments_directory / run_id / "metrics")
        for run_name, run_id in (
            (r"$\#\text{views}=1$", "bd44b1deccf94e68adea16a80de1265a"),
            (r"$\#\text{views}=2$", "5df00e3beccf42129379b0056d947dfa"),
            (r"$\#\text{views}=4$", "d924975a76ce476d93a635a1bee7b622"),
            (r"$\#\text{views}=8$", "f8e73fdc57ee4428b4caf43def4f1b29"),
            (r"$\#\text{views}=16$", "c1776255a9c34a22a8b351a0f9aabbe0"),
        )
    }

    plot(metrics)


if __name__ == "__main__":
    main()
