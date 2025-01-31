from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from scripts import MLFLOW_DIR, EXPORT_PATH
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

    axes[0].set_ylim(1.75, 8.25)
    axes[1].set_ylim(1.75, 8.25)

    axes[0].set_ylabel("Teacher Entropy")
    axes[1].set_ylabel("Student Entropy")
    axes[1].set_xlabel("Epoch")

    axes[0].legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "local_view_crop_scale_entropy.pdf")


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "246076156900597482"

    metrics: dict[str, pd.DataFrame] = {
        run_name: load_batch_metrics(experiments_directory / run_id / "metrics")
        for run_name, run_id in (
            (r"$\text{scale}\sim[0.05;0.10]$", "c364a9694bfd49f99ae728ac8abde348"),
            (r"$\text{scale}\sim[0.20;0.25]$", "10f4b94b896f4782b2f7f281b70d3f09"),
            (r"$\text{scale}\sim[0.35;0.40]$", "36c272adebcf43ab85bdbcdc4580021f"),
        )
    }

    plot(metrics)


if __name__ == "__main__":
    main()
