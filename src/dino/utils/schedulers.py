from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class Scheduler(ABC, Generic[T]):
    """Abstract class describing the functionalities of a scheduler."""

    @abstractmethod
    def get_value(self) -> T:
        """Returns the current value according to the schedule.

        :return: Current value according to the schedule.
        """

    @abstractmethod
    def step(self) -> None:
        """Function called to inform the scheduler to take a step in the schedule."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the schedule to the beginning."""


class ConstantScheduler(Scheduler[T]):
    def __init__(self, constant: T) -> None:
        self.constant = constant

    def get_value(self) -> T:
        return self.constant

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass


class CosineScheduler(Scheduler[float]):
    def __init__(self, initial: float, final: float, num_epochs: int, n_iters: int) -> None:
        """Initializes a CosineScheduler.

        :param initial: Initial value of the schedule.
        :param final: Final value of the schedule.
        :param num_epochs: Number of epochs to calculate the total steps.
        :param n_iters: Number of iterations per epoch to calculate the total steps.
        """
        self.initial = initial
        self.final = final
        self.current_step = 0
        self.total_steps = (num_epochs * n_iters) - 1

    def get_value(self) -> float:
        """Returns the current value according to a cosine schedule.

        :return: Value according to a cosine schedule.
        """
        value: float = self.final + (self.initial - self.final) * 0.5 * (
            1 + np.cos(np.pi * (self.current_step / self.total_steps))
        )
        return value

    def reset(self) -> None:
        """Resets the scheduler by setting the current step to 0."""
        self.current_step = 0

    def step(self) -> None:
        """Increases the step counter of the scheduler.

        If the step counter has reached the total number of steps, it cannot be increased any further.
        """
        if self.current_step < self.total_steps:
            self.current_step += 1
