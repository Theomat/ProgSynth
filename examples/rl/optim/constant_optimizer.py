from typing import Callable, List, Optional, Tuple

from optim.tiles import Tile, tile_split, find
from bandits.bandit import MultiArmedBandit

from synth.syntax.program import Constant


import numpy as np


class ConstantOptimizer:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.c = 0.7
        self.min_budget_per_arm = 15
        self.best_return = 0
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        eval: Callable[[], float],
        constants: List[Constant],
    ) -> Tuple[List[Tile], List[List[float]]]:
        self.budget_used = 0
        self._constants = constants
        tiles = [tile_split(-np.inf, np.inf, splits=4) for _ in constants]
        self._eval = eval
        self._can_hope_to_beat_best = True
        return self._optimize_tiles_(constants, tiles)

    def _pick_values(self) -> List[int]:
        arms = []
        for index, bandit in enumerate(self._bandits):
            arm = bandit.choose_arm_ucb()
            arms.append(arm)
            self._constants[index].assign(
                self._tiles_list[index][arm].map(self._rng.uniform(0, 1))
            )
        return arms

    def _update_with_return(
        self,
        arms: List[int],
        bandit_return: float,
        experiences: List[List[List[Tuple[float, float]]]],
    ):
        index = 0
        min_best_possible = 1e99
        for arm, bandit, experience in zip(arms, self._bandits, experiences):
            experience[arm].append((self._constants[index].value, bandit_return))
            index += 1
            bandit.add_return(arm, bandit_return)
            if (
                np.min(bandit._counts) > 0
                and bandit.best_possible_return() < min_best_possible
            ):
                min_best_possible = bandit.best_possible_return()
        self._can_hope_to_beat_best = min_best_possible >= self.best_return

    def _optimize_tiles_(
        self,
        constants: List[Constant],
        tiles_list: List[List[Tile]],
        prev_experiences=None,
        max_total_budget=1500,
    ) -> Tuple[List[Tile], List[List[float]]]:
        self._constants = constants
        self._tiles_list = tiles_list
        our_bandits = [MultiArmedBandit(len(tiles), self.c) for tiles in tiles_list]
        self._bandits = our_bandits

        experiences = [[[] for _ in tiles] for tiles in tiles_list]

        min_budget_to_use = (
            sum([bandit.arms for bandit in self._bandits]) * self.min_budget_per_arm
        )
        budget = max_total_budget
        budget_used = 0

        # Re use previous experience
        if prev_experiences:
            for bandit, experience, tiles in zip(
                self._bandits, prev_experiences, tiles_list
            ):
                for sampled_value, arm_return in experience:
                    arm = find(tiles, sampled_value)
                    bandit.add_return(arm, arm_return)
                    min_budget_to_use -= 1

        should_split = False
        has_used_min_budget = False

        # Do additional sampling while hope
        while not has_used_min_budget or (
            budget > 0 and self._can_hope_to_beat_best and not should_split
        ):
            # Assign values
            arms = self._pick_values()
            bandit_return = self._eval()
            self._update_with_return(arms, bandit_return, experiences)
            budget -= 1
            budget_used += 1
            has_used_min_budget = budget_used >= min_budget_to_use
            should_split = has_used_min_budget and budget > min_budget_to_use

        self.budget_used += budget_used

        # Compute best tiles
        best_tiles: List[Tile] = [
            tiles[bandit.best_arm()] for tiles, bandit in zip(tiles_list, self._bandits)
        ]
        # If better, then split
        if should_split:
            splits_size = [len(tiles) for tiles in tiles_list]
            next_tiles = [
                tile_split(best_tile.low, best_tile.high, splits)
                for best_tile, splits in zip(best_tiles, splits_size)
            ]
            # Now we actually might discard some constants if we believe we cannot optimise them anymore
            indices_to_keep = [
                i for i in range(len(constants)) if best_tiles[i].size() > 1e-2
            ]
            if len(indices_to_keep) > 0:
                constants = [c for i, c in enumerate(constants) if i in indices_to_keep]
                next_tiles = [
                    c for i, c in enumerate(next_tiles) if i in indices_to_keep
                ]
                new_experiences = [
                    xp[bandit.best_arm()]
                    for i, (xp, bandit) in enumerate(zip(experiences, self._bandits))
                    if i in indices_to_keep
                ]
                a, b = self._optimize_tiles_(
                    constants,
                    next_tiles,
                    max_total_budget=budget,
                    prev_experiences=new_experiences,
                )
                # Merge back all tiles
                all_tiles = []
                j = 0
                for i, tile in enumerate(best_tiles):
                    if i in indices_to_keep:
                        all_tiles.append(a[j])
                        j += 1
                    else:
                        all_tiles.append(tile)
                return all_tiles, b

        returns = []
        for bandit in our_bandits:
            if len(bandit.returns[bandit.best_arm()]) > len(returns):
                returns = bandit.returns[bandit.best_arm()]
        return best_tiles, returns
