import numpy as np

from typing import List, Tuple


class MultiArmedBandit:
    def __init__(self, arms: int, c: float = 0.6) -> None:
        self.arms: int = arms
        self.c: float = c
        self.returns: List[List[float]] = [[] for _ in range(self.arms)]
        self._mean_returns: List[float] = [0 for _ in range(self.arms)]
        self._counts: List[int] = [0 for _ in range(self.arms)]
        self._time: int = 0
        self._best_arm: int = 0

    def samples(self, arm: int) -> int:
        return self._counts[arm]

    def reset_arm(self, arm: int):
        self.returns[arm].clear()
        self._time -= self._counts[arm]
        self._counts[arm] = 0
        if arm == self.best_arm:
            self._mean_returns[arm] = -np.inf
            self._best_arm = np.argmax(self._mean_returns)
        self._mean_returns[arm] = 0

    def add_return(self, arm: int, arm_return: float) -> None:
        self.returns[arm].append(arm_return)
        self._counts[arm] += 1
        mean_increase: float = (arm_return - self._mean_returns[arm]) / self._counts[
            arm
        ]
        self._mean_returns[arm] += mean_increase
        self._time += 1
        if arm == self._best_arm:
            if mean_increase < 0:
                self._best_arm = np.argmax(self._mean_returns)
        elif self._mean_returns[arm] > self._mean_returns[self._best_arm]:
            self._best_arm = arm

    def choose_arm_ucb(self) -> int:
        if np.min(self._counts) == 0:
            return np.argmin(self._counts)
        incertitudes = [
            self.c * np.sqrt(np.log(self._time) / count) for count in self._counts
        ]
        best_returns = [
            self._mean_returns[i] + incertitudes[i] for i in range(self.arms)
        ]
        return np.argmax(best_returns)

    def best_arm(self) -> int:
        return self._best_arm

    def best_return(self) -> float:
        return (
            self._mean_returns[self._best_arm]
            if self._counts[self._best_arm]
            else np.inf
        )

    def worst_arm(self) -> int:
        return np.argmin(
            [
                self._mean_returns[i] if self._counts[i] else np.inf
                for i in range(self.arms)
            ]
        )

    def incertitude(self, arm: int) -> float:
        if self._counts[arm]:
            return self.c * np.sqrt(np.log(self._time) / self._counts[arm])
        return np.inf

    def possible_returns(self) -> List[Tuple[float, float]]:
        incertitudes = [self.incertitude(arm) for arm in range(self.arms)]
        return [
            (q_value - inc, q_value + inc)
            for q_value, inc in zip(self._mean_returns, incertitudes)
        ]

    def best_possible_return(self) -> float:
        assert np.min(self._counts) > 0
        incertitudes = [
            self.c * np.sqrt(np.log(self._time) / count) for count in self._counts
        ]
        best_returns = [
            self._mean_returns[i] + incertitudes[i] for i in range(self.arms)
        ]
        return np.max(best_returns)
