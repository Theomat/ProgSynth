from typing import Callable, List, Optional, Tuple

from rl.rl_utils import get_returns

from synth.syntax import Program
from synth.semantic.evaluator import Evaluator

import gymnasium as gym

import numpy as np


def __state2env__(state: np.ndarray) -> Tuple:
    return tuple(state.tolist())


def __adapt_action2env__(env: gym.Env, action) -> List:
    if isinstance(env.action_space, gym.spaces.Box):
        if len(env.action_space.shape) == 1 and env.action_space.shape[0] == 1:
            return [action]
    return action


def eval_function(
    env: gym.Env, evaluator: Evaluator
) -> Callable[[Program], Tuple[bool, float]]:
    def func(
        program: Program, state: List[int] = None, debug: bool = False
    ) -> Tuple[bool, float]:
        success, episodes = eval_program(env, program, evaluator, 1, state, debug=debug)
        returns = get_returns(episodes)[0]
        return (
            (success, returns)
            if success
            else ((0, episodes[0]) if success else (None, None))
        )

    return func


def eval_program(
    env: gym.Env,
    program: Program,
    evaluator: Evaluator,
    n: int,
    state: List[int] = None,
    debug: bool = False,
) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    episodes = []
    try:
        for _ in range(n):
            episode = []
            if state is None:
                state = env.reset()[0]
            else:
                env.reset()[0]
                state = env.unwrapped.set_state_deterministic(state)[0]
            done = False
            while not done:
                input = __state2env__(state)
                action = evaluator.eval(program, input)
                adapted_action = __adapt_action2env__(env, action)
                if adapted_action not in env.action_space:
                    return False, None
                next_state, reward, done, truncated, _ = env.step(adapted_action)
                done |= truncated
                episode.append((state.copy(), action, reward))
                state = next_state
            episodes.append(episode)
    except OverflowError:
        return False, None
    return True, episodes


def render_program(env: gym.Env, program: Program, evaluator: Evaluator, record=True):
    if record:
        env = gym.wrappers.RecordVideo(
            env, "./videos", episode_trigger=lambda p: True, disable_logger=True
        )
    state = env.reset()
    done = False
    while not done:
        env.render()
        input = __state2env__(state)
        action = evaluator.eval(program, input)
        adapted_action = __adapt_action2env__(env, action)
        next_state, _, done, truncated, _ = env.step(adapted_action)
        done |= truncated
        state = next_state
    env.close()
