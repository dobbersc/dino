import functools
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.lines import Line2D


def load_epoch_metrics(metrics_directory: Path) -> pd.DataFrame:
    metrics: list[pd.DataFrame] = [
        pd.read_csv(
            metrics_directory / f"train_epoch_{metric_name}",
            sep=" ",
            usecols=(1, 2),
            index_col="epoch",
            names=(metric_name, "epoch"),
        )
        for metric_name in (
            "loss",
            "student_entropy",
            "teacher_entropy",
            "kl_divergence",
            "student_accuracy",
            "teacher_accuracy",
        )
    ]
    return functools.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), metrics)


def load_batch_metrics(metrics_directory: Path) -> pd.DataFrame:
    metrics: list[pd.DataFrame] = [
        pd.read_csv(
            metrics_directory / f"train_batch_{metric_name}",
            sep=" ",
            usecols=(1, 2),
            index_col="batch",
            names=(metric_name, "batch"),
        )
        for metric_name in (
            "epoch",
            "loss",
            "student_entropy",
            "teacher_entropy",
            "kl_divergence",
            "learning_rate",
            "weight_decay",
            "center_momentum",
            "teacher_momentum",
            "teacher_temperature",
            "student_temperature",
        )
    ]
    return functools.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), metrics)


def plot_error_tube(
    line: Line2D,
    error: pd.Series,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
    alpha: float = 0.25,
    **kwargs: Any,
) -> None:
    y = line.get_ydata()
    line.axes.fill_between(
        line.get_xdata(),
        (y - error).clip(lower=clip_lower),
        (y + error).clip(upper=clip_upper),
        color=line.get_color(),
        alpha=alpha,
        **kwargs,
    )
