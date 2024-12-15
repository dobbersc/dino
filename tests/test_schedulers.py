from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import pytest

from dino.utils.schedulers import ConstantScheduler, LinearScheduler, Scheduler, SequentialScheduler

_T = TypeVar("_T")


def _build_test_sequential_scheduler(milestones: Sequence[int]) -> SequentialScheduler[float]:
    return SequentialScheduler(
        [
            LinearScheduler(initial=0.0, final=5.0, max_steps=6),
            LinearScheduler(initial=10.0, final=7.0, max_steps=4),
            LinearScheduler(initial=0.0, final=3.0, max_steps=4),
        ],
        milestones=milestones,
    )


@pytest.mark.parametrize(
    ("scheduler", "expected_values"),
    [
        (ConstantScheduler(constant=1.0, max_steps=10), [1.0] * 11),
        (LinearScheduler(initial=0.0, final=10.0, max_steps=11), np.arange(0.0, 12.0).tolist()),
        (LinearScheduler(initial=-5.0, final=-10.0, max_steps=6), np.arange(-11.0, -4.0)[::-1].tolist()),
        (
            _build_test_sequential_scheduler(milestones=[6, 10]),
            np.concatenate((np.arange(0.0, 6.0), np.arange(7.0, 11.0)[::-1], np.arange(0.0, 5.0))).tolist(),
        ),
    ],
    ids=[
        f"{ConstantScheduler.__name__}",
        f"{LinearScheduler.__name__}_increasing",
        f"{LinearScheduler.__name__}_decreasing",
        f"{SequentialScheduler.__name__}",
    ],
)
def test_scheduler(scheduler: Scheduler[_T], expected_values: list[_T]) -> None:
    assert scheduler.max_steps is not None

    values: list[_T] = []
    for _ in range(scheduler.max_steps):
        values.append(scheduler.get_value())
        scheduler.step()
    values.append(scheduler.get_value())

    assert values == expected_values

    with pytest.raises(
        RuntimeError,
        match=rf"Exceeded the maximum number of steps \({scheduler.max_steps}\) supported by the scheduler.",
    ):
        scheduler.step()


class TestSequentialScheduler:

    def test_invalid_number_of_milestones(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                r"SequentialScheduler expects the number of schedulers provided "
                r"to be one more than the number of milestone points."
            ),
        ):
            _build_test_sequential_scheduler(milestones=[1, 2, 3])

    def test_exceeding_milestones(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                r"Scheduler at index 1 starting at milestone 6 and "
                r"ending at before 12 exceeds its maximum number of steps \(4\) by 1 step\(s\)."
            ),
        ):
            _build_test_sequential_scheduler(milestones=[6, 12])
