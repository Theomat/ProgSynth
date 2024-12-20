from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Tile:
    low: float
    high: float

    def map(self, x: float) -> float:
        if np.isneginf(self.low):
            return -1e10 + (self.high - -1e10) * x
        elif np.isposinf(self.high):
            return self.low + (1e10 - self.low) * x
        return self.low + (self.high - self.low) * x

    def __repr__(self) -> str:
        return f"[{self.low};{self.high}]"

    def size(self) -> float:
        return self.high - self.low

    def midpoint(self) -> float:
        return (self.high + self.low) / 2


def tile_split(
    low: float, high: float, splits: int = 12, critical_tile_size: float = 20
) -> List[Tile]:
    tiles: List[Tile] = []

    # ====================
    if np.isinf(low):
        low = -1e10
    if np.isinf(high):
        high = 1e10
    # ====================

    if -low == high:
        # Symetric split
        half_splits = splits // 2
        if np.isneginf(low) or high > 10 ** (half_splits - 1):
            start = low
            for i in reversed(range(half_splits - 1)):
                tiles.append(Tile(start, -(10**i)))
                start = tiles[-1].high
            tiles.append(Tile(-1, 0))
            start = 0
            for i in range(half_splits - 1):
                tiles.append(Tile(start, 10**i))
                start = tiles[-1].high
            tiles.append(Tile(start, high))
        else:
            if (high - low) / splits > critical_tile_size:
                # Logarithmic scale
                start = low
                powers = []
                log_scale = splits / abs(np.log10(np.abs(high)) + np.log10(np.abs(low)))
                power = np.log10(np.abs(start))
                for i in reversed(range(half_splits - 1)):
                    power += np.sign(start) / log_scale
                    powers.append(power)
                    tiles.append(Tile(start, -(10**power)))
                    start = tiles[-1].high
                tiles.append(Tile(start, 0))
                start = 0
                for power in reversed(powers):
                    tiles.append(Tile(start, 10**power))
                    start = tiles[-1].high
                tiles.append(Tile(start, high))
            else:
                start = low
                dt = (high - low) / splits
                for x in np.arange(low + dt, high + dt / 2, dt):
                    tiles.append(Tile(start, x))
                    start = tiles[-1].high
    else:
        if np.sign(low) == np.sign(high) or low == 0 or high == 0:
            if (high - low) / splits > critical_tile_size:
                # Check infinity
                if np.isposinf(high):
                    tiles.append(Tile(max(low, 1) * 10**splits, high))
                    high = tiles[-1].low
                    splits -= 1
                elif np.isneginf(low):
                    tiles.append(Tile(low, min(high, -1) * 10**splits))
                    low = tiles[-1].high
                    splits -= 1
                # Logarithmic scale
                start = low
                log_scale = splits / abs(np.log10(np.abs(high)) - np.log10(np.abs(low)))
                power = np.log10(np.abs(start))
                for i in range(splits - 1):
                    power += np.sign(start) / log_scale
                    tiles.append(Tile(start, np.sign(start) * 10**power))
                    start = tiles[-1].high
                tiles.append(Tile(start, high))
            else:
                start = low
                dt = (high - low) / splits
                if dt <= 0:
                    tiles.append(Tile(low, high))
                else:
                    for x in np.linspace(low + dt, high, splits):
                        tiles.append(Tile(start, x))
                        start = tiles[-1].high
        else:
            # Complicated case so just cheat kind of
            positive_splits = int(np.round(high / (high - low) * splits))
            if positive_splits == splits:
                positive_splits -= 1
            return tile_split(low, 0, splits - positive_splits) + tile_split(
                0, high, positive_splits
            )
    return tiles


def find(tiles: List[Tile], x: float) -> int:
    for i, tile in enumerate(tiles):
        if tile.low <= x and tile.high >= x:
            return i
    assert False


def sample(
    tiles: List[Tile],
    probabilities: Union[List[float], np.ndarray],
    generator: np.random.Generator,
) -> Tuple[float, Tile]:
    tile: Tile = generator.choice(tiles, p=probabilities)
    x = generator.uniform(0, 1)
    return tile.map(x), tile
