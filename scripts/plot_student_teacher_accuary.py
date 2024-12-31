from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from scripts import EXPORT_PATH, MLFLOW_DIR

if TYPE_CHECKING:
    from pathlib import Path


def plot(student_accuracies: pd.DataFrame, teacher_accuracies: pd.DataFrame) -> None:
    figure, ax = plt.subplots()

    ax.plot(
        student_accuracies["epoch"],
        student_accuracies["accuracy"] * 100,
        alpha=0.75,
        linestyle="--",
        label="Student",
    )
    ax.plot(
        teacher_accuracies["epoch"],
        teacher_accuracies["accuracy"] * 100,
        alpha=0.75,
        label="Teacher",
    )

    ax.set_ylim(0, 100)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.legend()

    figure.tight_layout()
    figure.show()

    if EXPORT_PATH is not None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        figure.savefig(EXPORT_PATH / "deit_student_teacher_accuracy.pdf")


def main() -> None:
    # Path for DeiT-Small architecture run.
    metrics_directory: Path = MLFLOW_DIR / "465212383374859638/764f5becec7f4d0c9cb6c4289fe646e8/metrics"

    student_accuracies: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_epoch_student_accuracy",
        sep=" ",
        usecols=(1, 2),
        names=("accuracy", "epoch"),
    )
    teacher_accuracies: pd.DataFrame = pd.read_csv(
        metrics_directory / "train_epoch_teacher_accuracy",
        sep=" ",
        usecols=(1, 2),
        names=("accuracy", "epoch"),
    )

    plot(student_accuracies=student_accuracies, teacher_accuracies=teacher_accuracies)


if __name__ == "__main__":
    main()
