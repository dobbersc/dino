from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from scripts import EXPORT_PATH, MLFLOW_DIR


def plot(sharpening_and_centering: pd.DataFrame, no_sharpening: pd.DataFrame, no_centering: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, sharex=True)

    axes[0].plot(
        sharpening_and_centering["epoch"],
        sharpening_and_centering["entropy"],
        alpha=0.75,
        label="Sharpening & Centering",
    )
    axes[0].plot(
        no_sharpening["epoch"],
        no_sharpening["entropy"],
        alpha=0.75,
        linestyle="--",
        label="No Sharpening",
    )
    axes[0].plot(
        no_centering["epoch"],
        no_centering["entropy"],
        alpha=0.6,
        label="No Centering",
    )

    axes[1].plot(
        sharpening_and_centering["epoch"],
        sharpening_and_centering["kl_divergence"],
        alpha=0.75,
        label="Sharpening & Centering",
    )
    axes[1].plot(
        no_sharpening["epoch"],
        no_sharpening["kl_divergence"],
        alpha=0.75,
        linestyle="--",
        label="No Sharpening",
    )
    axes[1].plot(
        no_centering["epoch"],
        no_centering["kl_divergence"],
        alpha=0.6,
        label="No Centering",
    )

    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))
    axes[1].yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))

    axes[0].set_ylabel("Teacher Entropy")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_xlabel("Epoch")

    axes[0].legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "model_collapse_entropy_kl_divergence.pdf")


def load_metrics(metric_directory: Path) -> pd.DataFrame:
    entropies: pd.DataFrame = pd.read_csv(
        metric_directory / "train_epoch_teacher_entropy",
        sep=" ",
        usecols=(1, 2),
        names=("entropy", "epoch"),
    )
    kl_divergences: pd.DataFrame = pd.read_csv(
        metric_directory / "train_epoch_kl_divergence",
        sep=" ",
        usecols=(1, 2),
        names=("kl_divergence", "epoch"),
    )
    return entropies.merge(kl_divergences, how="outer", on="epoch")


def main() -> None:
    experiments_directory: Path = MLFLOW_DIR / "465212383374859638"

    plot(
        # Metrics corresponding to run "architecture-deit-small"
        sharpening_and_centering=load_metrics(experiments_directory / "764f5becec7f4d0c9cb6c4289fe646e8/metrics"),
        # Metrics corresponding to run "model-collapse-no-sharpening"
        no_sharpening=load_metrics(experiments_directory / "71bb329c920f495a8564abe2d3cfe1ee/metrics"),
        # Metrics corresponding to run "model-collapse-no-centering"
        no_centering=load_metrics(experiments_directory / "eb04f3af130e485a9277a17f51d0f0d0/metrics"),
    )


if __name__ == "__main__":
    main()
