from numpy import isposinf
from bandits.bandit import MultiArmedBandit

from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Generic


T = TypeVar("T", covariant=True)


class CleverEvaluator(Generic[T]):
    # Note that unlike the usual bandit we do not want to determine the best candidate rather we want to determine the worst candidate
    # Hence we negate the usual returns
    def __init__(
        self,
        get_return: Callable[[T], Tuple[bool, float]],
        candidates: int = 2,
        c: float = 1,
    ) -> None:
        self.bandit: MultiArmedBandit = MultiArmedBandit(candidates, c)
        self.get_return: Callable[[T], float] = get_return
        self.candidates: Dict[T, int] = {}
        self._arm2candidate: List[T] = []
        self._last_ejected: int = -1

    def num_candidates(self) -> int:
        return len(self.candidates)

    def challenge_with(
        self,
        new_candidate: T,
        max_budget: int = 100,
        prior_experience: List[float] = [],
    ) -> Tuple[Optional[T], int]:
        """
        return: the T ejected and the no of calls to get_return
        """
        # Add new program
        for arm in range(self.bandit.arms):
            if arm not in self.candidates.values():
                self.candidates[new_candidate] = arm
                self._arm2candidate.insert(arm, new_candidate)
                # Add prior experience
                for arm_return in prior_experience:
                    self.bandit.add_return(arm, -arm_return)
                break

        assert new_candidate in self.candidates
        if len(self._arm2candidate) < self.bandit.arms:
            return None, 0
        ejected_candidate, budget_used = self.__run_until_ejection__(max_budget)
        if ejected_candidate:
            self.__eject__(ejected_candidate)
        return ejected_candidate, budget_used

    def get_best_stats(self) -> Tuple[T, float, float, float, float]:
        arm: int = self.bandit.worst_arm()
        candidate = (
            self._arm2candidate[arm]
            if arm < self._last_ejected
            else self._arm2candidate[arm - 1]
        )
        best_returns = [-x for x in self.bandit.returns[arm]]
        if len(best_returns) == 0:
            return candidate, float("nan"), float("inf"), -float("inf"), float("inf")
        mean_return = sum(best_returns) / len(best_returns)
        return (
            candidate,
            mean_return,
            self.bandit.incertitude(arm),
            min(best_returns),
            max(best_returns),
        )

    def run_at_least(self, min_budget: int) -> int:
        arm: int = self.bandit.worst_arm()
        candidate = (
            self._arm2candidate[arm]
            if arm < self._last_ejected
            else self._arm2candidate[arm - 1]
        )
        budget_used: int = 0
        while self.bandit.samples(arm) < min_budget:
            can_continue, arm_return = self.get_return(candidate)
            budget_used += 1
            if not can_continue:
                break
            self.bandit.add_return(arm, -arm_return)
        return budget_used

    def __run_until_ejection__(self, max_budget: int) -> Tuple[Optional[T], int]:
        """
        return: the T ejected and the cost
        """
        budget_used: int = 0
        while self.__get_candidate_to_eject__() is None and budget_used < max_budget:
            arm: int = self.bandit.choose_arm_ucb()
            candidate: T = self._arm2candidate[arm]
            can_continue, arm_return = self.get_return(candidate)
            if not can_continue:
                return candidate, budget_used
            self.bandit.add_return(arm, -arm_return)
            budget_used += 1
        return self.__get_candidate_to_eject__(True), budget_used

    def __get_candidate_to_eject__(self, force: bool = False) -> Optional[T]:
        worst_arm = self.bandit.best_arm()
        if force:
            return self._arm2candidate[worst_arm]
        return_intervals = self.bandit.possible_returns()
        low, high = return_intervals.pop(worst_arm)
        midpoint: float = (high + low) / 2
        if not isposinf(high):
            for other_low, other_high in return_intervals:
                if midpoint >= other_low and midpoint <= other_high:
                    return None
            return self._arm2candidate[worst_arm]
        return None

    def __eject__(self, candidate: T):
        arm: int = self.candidates[candidate]
        self.bandit.reset_arm(arm)

        self._last_ejected: int = arm

        del self._arm2candidate[arm]
        del self.candidates[candidate]
