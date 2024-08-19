from typing import Optional, Union
import numpy as np


class PythonSampler:
    def __init__(self, weights: np.ndarray, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed or 1)
        n = len(weights)
        alias = np.zeros(n, dtype=int)
        proba = np.zeros(n, dtype=float)
        # Compute the average probability and cache it for later use.
        avg = 1.0 / n
        # Create two stacks to act as worklists as we populate the tables.
        small = []
        large = []
        # Populate the stacks with the input probabilities.
        for i in range(n):
            # If the probability is below the average probability, then we add it to the small
            # list; otherwise we add it to the large list.
            if weights[i] >= avg:
                large.append(i)
            else:
                small.append(i)
        # As a note: in the mathematical specification of the algorithm, we will always exhaust the
        # small list before the big list. However, due to floating point inaccuracies, this is not
        # necessarily true. Consequently, this inner loop (which tries to pair small and large
        # elements) will have to check that both lists aren't empty.
        while len(small) > 0 and len(large) > 0:
            # Get the index of the small and the large probabilities.
            less = small.pop(0)
            more = large.pop(0)
            # These probabilities have not yet been scaled up to be such that 1 / n is given weight
            # 1.0. We do this here instead.
            proba[less] = weights[less] * n
            alias[less] = more
            # Decrease the probability of the larger one by the appropriate amount.
            weights[more] = weights[more] + weights[less] - avg
            # If the new probability is less than the average, add it into the small list;
            # otherwise add it to the large list.
            if weights[more] >= avg:
                large.append(more)
            else:
                small.append(more)
        # At this point, everything is in one list, which means that the remaining probabilities
        # should all be 1 / n.  Based on this, set them appropriately. Due to numerical issues, we
        # can't be sure which stack will hold the entries, so we empty both.
        while len(small) > 0:
            less = small.pop(0)
            proba[less] = 1.0
        while len(large) > 0:
            more = large.pop(0)
            proba[more] = 1.0
        self.n = n
        self.alias = alias
        self.proba = proba

    def sample_1(self) -> int:
        # Generate a fair die roll to determine which column to inspect.
        col = int(self.rng.uniform(0, self.n))
        # Generate a biased coin toss to determine which option to pick.
        heads = self.rng.uniform() < 0.5

        # Based on the outcome, return either the column or its alias.
        if heads:
            return col
        return self.alias[col]  # type: ignore

    def sample(
        self, k: int = 1, values: Optional[np.ndarray] = None
    ) -> Union[int, np.ndarray]:
        """Sample a random integer or a value from a given array.

        Parameters:
            k: The number of integers to sample. If `k = 1`, then a single int (or float if values is not None) is returned. In any
                other case, a numpy array is returned.
            values: The numpy array of values from which to sample from.

        """
        if values is None:
            if k == 1:
                return self.sample_1()
            return np.asarray([self.sample_1() for _ in range(k)])
        else:
            if k == 1:
                return values[self.sample_1()]  # type: ignore
            return np.asarray([values[self.sample_1()] for _ in range(k)])


try:
    import vose

    Sampler = vose.Sampler
except ImportError:
    Sampler = PythonSampler
