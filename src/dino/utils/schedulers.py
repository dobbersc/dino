from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class Scheduler(ABC, Generic[T]):
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
    def get_value(self) -> T:
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


class ConstantScheduler(Scheduler[T]):
    def __init__(self, constant: T) -> None:
        super().__init__(max_steps=None)

        self.constant = constant

    def get_value(self) -> T:
        return self.constant


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
