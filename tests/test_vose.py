import vose

import numpy as np


def test_seeding() -> None:
    for _ in range(100):
        probs = np.random.randn((10))
        probs /= np.sum(probs)
        seed = np.random.randint(9999999)
        a = vose.Sampler(probs, seed=seed)
        b = vose.Sampler(probs, seed=seed)
        for i in range(100):
            assert a.sample() == b.sample()
