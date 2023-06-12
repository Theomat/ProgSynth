from typing import Optional
from karel_runtime import KarelWorld

import numpy as np


def random_world(
    width: int, height: int, rng: Optional[np.random.Generator] = None
) -> KarelWorld:
    world = KarelWorld(width, height)
    gen = rng or np.random.default_rng()
    world.grid[gen.random((width, height)) > 0.8] = 1
    world.markers = (gen.random((width, height)) > 0.7).astype(int)
    world.markers[world.grid] = 0
    return world
