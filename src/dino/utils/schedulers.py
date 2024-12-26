import itertools
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_T = TypeVar("_T")


class Scheduler(ABC, Generic[_T]):
    """Abstract class describing the functionalities of a scheduler."""

    def __init__(self, max_steps: int | None) -> None:
        """Initializes a Scheduler.

        Args:
            max_steps: The maximum number of steps supported by the scheduler.
                If None, the scheduler has no step limit.
        """
        self._max_steps = max_steps
        self._current_step: int = 0

    @abstractmethod
    def get_value(self) -> _T:
        """Returns the current values according to the schedule."""

    def step(self) -> None:
        """Informs the scheduler to take a step in the schedule."""
        if self._max_steps is not None and self._current_step == self._max_steps:
            msg: str = f"Exceeded the maximum number of steps ({self.max_steps}) supported by the scheduler."
            raise RuntimeError(msg)

        self._current_step += 1

    def reset(self) -> None:
        """Resets the schedule to the beginning."""
        self._current_step = 0

    @property
    def max_steps(self) -> int | None:
        return self._max_steps

    @property
    def current_step(self) -> int:
        return self._current_step

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_steps={self.max_steps!r})"


class SequentialScheduler(Scheduler[_T]):
    """Utility scheduler that enables the sequential execution of multiple schedulers.

    Milestone points control the exact intervals when the scheduler is active.
    """

    def __init__(self, schedulers: Sequence[Scheduler[_T]], milestones: Sequence[int]) -> None:
        """Initializes a SequentialScheduler.

        Args:
            schedulers: A sequence of chained schedulers.
            milestones: A sequence of milestones specifying the step after which the next scheduler in the sequence will
             continue. A scheduler will be active when the step lies in the interval
             [previous_milestone, current_milestone). For the first scheduler, `previous_milestone = 1` is assumed.
             Note that the number of schedulers provided must be one more than the number of milestone points
             to ensure a well-defined schedule.
        """
        self.schedulers = schedulers
        self.milestones = sorted(milestones)

        super().__init__(
            max_steps=(
                None if self.schedulers[-1].max_steps is None else self.milestones[-1] + self.schedulers[-1].max_steps
            ),
        )

        if len(self.milestones) != len(self.schedulers) - 1:
            msg: str = (
                "SequentialScheduler expects the number of schedulers provided to be one more than the number of "
                f"milestone points. But got number of schedulers {len(self.schedulers)} and "
                f"number of milestones {len(self.milestones)}."
            )
            raise ValueError(msg)

        for (index, scheduler), (previous_milestone, milestone) in zip(
            enumerate(self.schedulers[:-1]),
            itertools.pairwise(itertools.chain([0], self.milestones)),
            strict=True,
        ):
            if scheduler.max_steps is not None and scheduler.max_steps < milestone - previous_milestone - 1:
                msg = (
                    f"Scheduler at index {index} starting at milestone {previous_milestone} and "
                    f"ending at before {milestone} exceeds its maximum number of steps ({scheduler.max_steps}) "
                    f"by {milestone - previous_milestone - scheduler.max_steps - 1} step(s)."
                )
                raise ValueError(msg)

    def step(self) -> None:
        super().step()

        # Only perform a step on the current scheduler if the initial value has been part of the value sequence.
        # This is the case when the current step is not a milestone due to the schedulers' left-open activity intervals.
        if self.current_step not in self.milestones:
            self.current_scheduler.step()

    def get_value(self) -> _T:
        return self.current_scheduler.get_value()

    @property
    def current_scheduler(self) -> Scheduler[_T]:
        index: int = bisect_right(self.milestones, self.current_step)
        return self.schedulers[index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(schedulers={self.schedulers!r}, milestones={self.milestones!r})"


class ConstantScheduler(Scheduler[_T]):
    def __init__(self, constant: _T, max_steps: int | None = None) -> None:
        super().__init__(max_steps=max_steps)

        self.constant = constant

    def get_value(self) -> _T:
        return self.constant


class LinearScheduler(Scheduler[float]):
    def __init__(self, max_steps: int, initial: float, final: float) -> None:
        """Initializes a LinearScheduler.

        Args:
            max_steps: The maximum number of steps supported by the scheduler.
            initial: The initial value of the schedule.
            final: The final value of the schedule.
                Note that the scheduler reaches the linear interpolation's final value at step `max_steps - 1`.
        """
        super().__init__(max_steps=max_steps)

        self.initial = initial
        self.final = final

    def get_value(self) -> float:
        assert self._max_steps is not None
        t_max: int = self._max_steps - 1
        offset: float = (self.final - self.initial) * (self.current_step / t_max)
        return self.initial + offset

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_steps={self.max_steps!r}, initial={self.initial!r}, final={self.final!r})"


class CosineScheduler(Scheduler[float]):
    def __init__(self, max_steps: int, initial: float, final: float) -> None:
        """Initializes a CosineScheduler.

        Args:
            max_steps: The maximum number of steps supported by the scheduler.
            initial: The initial value of the schedule.
            final: The final value of the schedule.
                Note that the scheduler reaches the cosine interpolation's final value at step `max_steps - 1`.
        """
        super().__init__(max_steps=max_steps)

        self.initial = initial
        self.final = final

    def get_value(self) -> float:
        assert self._max_steps is not None
        t_max: int = self._max_steps - 1
        offset: NDArray[np.float64] = (
            0.5 * (self.initial - self.final) * (1 + np.cos(np.pi * (self._current_step / t_max)))
        )
        return self.final + offset.item()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_steps={self.max_steps!r}, initial={self.initial!r}, final={self.final!r})"
