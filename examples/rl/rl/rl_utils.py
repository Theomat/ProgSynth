from typing import Any, List, Tuple

from gymnasium.core import Env

from synth.syntax import auto_type

import numpy as np

import gymnasium as gym

from synth.syntax.type_system import Type

ACTION = auto_type("ACTION")
FLOAT = auto_type("float")


def get_returns(episodes: List[List[Tuple[np.ndarray, int, float]]]) -> List[float]:
    return [sum([trans[2] for trans in episode]) for episode in episodes]


def stats(returns: List[float]) -> Tuple[float, float, float, float, float]:
    return (
        np.min(returns),
        np.max(returns),
        np.mean(returns),
        np.std(returns),
        np.median(returns),
    )


def __compute_derivative__(
    low: float | np.ndarray, high: float | np.ndarray, shape: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(low, float):
        low = np.ones(shape, dtype=float) * low
    if isinstance(high, float):
        high = np.ones(shape, dtype=float) * high
    der_low = low - high
    der_high = high - low
    return np.stack([low, der_low]).flatten(), np.stack([high, der_high]).flatten()


def __add_derivative__(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Box):
        new_low, new_high = __compute_derivative__(space.low, space.high, space.shape)
        shape = list(space.shape)
        shape[-1] *= 2
        return gym.spaces.Box(
            new_low, new_high, tuple(shape), dtype=space.dtype, seed=space.np_random
        )
    else:
        assert False, f"Unsupported space type:{space}"


class DerivativeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env, derivative_step: int = 1):
        super().__init__(env)
        self.derivative_step = derivative_step
        self._last_obs = []
        self.observation_space = __add_derivative__(env.observation_space)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._last_obs = []
        return super().reset(seed=seed, options=options)

    def observation(self, observation: Any) -> Any:
        self._last_obs.append(observation)
        if len(self._last_obs) > 1 + self.derivative_step:
            self._last_obs.pop(0)
        out = [self._last_obs[-1], self._last_obs[-1] - self._last_obs[0]]
        return np.stack(out).flatten()


def type_for_env(env: gym.Env) -> Type:
    output = None
    if isinstance(env.action_space, gym.spaces.Discrete):
        n = env.action_space.n
        output = "action"
    elif isinstance(env.action_space, gym.spaces.Box):
        shape = env.action_space.shape
        if len(shape) == 1:
            if shape[0] == 1:
                output = "float"
            else:
                output = f"float {shape[0]}-tuple"
        else:
            assert False, "Can't handle multidimensional outputs!"

    if isinstance(env.observation_space, gym.spaces.Discrete):
        n = env.action_space.n
        if n == 2:
            return auto_type(f"bool->{output}")
        else:
            return auto_type(f"int->{output}")
    elif isinstance(env.observation_space, gym.spaces.Box):
        shape = env.observation_space.shape
        if len(shape) == 1:
            t = auto_type("->".join(["float"] * shape[0]) + "->" + output)
            return t
        assert False, "Can't handle multidimensional inputs!"
