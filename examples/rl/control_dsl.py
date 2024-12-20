from typing import Tuple

from synth.syntax import auto_type, DSL, Type, Constant
from synth.semantic import DSLEvaluator, Evaluator
from synth.filter.local_stateless_filter import reject_functions, commutative_rejection


import numpy as np
import gymnasium as gym


def __tuple_semantic(current: np.ndarray, value, remaining: int):
    current[-remaining] = value
    if remaining == 1:
        return current
    else:
        return lambda p: __tuple_semantic(current, p, remaining - 1)


def get_dsl(
    type_request: Type,
    output_space: gym.spaces.Space,
) -> Tuple[DSL, Evaluator]:
    primitive_types = auto_type(
        {
            "+": "float -> float -> float",
            "-": "float-> float",
            "*": "float -> float -> float",
            # "1/x": "float->float",
            ">=": "float->float->bool",
            "and": "bool->bool->bool",
            "or": "bool->bool->bool",
            "not": "bool->bool",
            # "log": "float->float",
            # "exp": "float->float",
            "sign": "float->float",
            "ite": "bool -> 'a[float|action] -> 'a[float|action] -> 'a[float|action]",
        }
    )

    semantics = {
        "+": lambda float1: lambda float2: float1 + float2,
        "-": lambda float: -float,
        "*": lambda float1: lambda float2: float1 * float2,
        "1/x": lambda float: 1 / float,
        ">=": lambda float1: lambda float2: float1 >= float2,
        "and": lambda bool1: lambda bool2: bool1 and bool2,
        "or": lambda bool1: lambda bool2: bool1 or bool2,
        "not": lambda bool: not bool,
        "log": np.log,
        "exp": np.exp,
        "sign": np.sign,
        "ite": lambda cond: lambda if_block: lambda else_block: if_block
        if cond
        else else_block,
    }

    # Handle discrete action space
    ACTION = auto_type("action")

    def add_actions(n: int):
        for i in range(n):
            primitive_types[f"A{i}"] = ACTION
            semantics[f"A{i}"] = i

    if type_request.ends_with(ACTION):
        add_actions(output_space.n)

    # Smart Filte

    is_useless = {
        "+": commutative_rejection,
        "*": commutative_rejection,
        "ite": lambda cond, p1, p2: reject_functions(cond, "not")
        or p1 == p2
        or (isinstance(p1, Constant) and isinstance(p2, Constant)),
        ">=": lambda p1, p2: p1 == p2
        or (isinstance(p1, Constant) and isinstance(p2, Constant)),
        "and": lambda p1, p2: hash(p1) >= hash(p2),
        "or": lambda p1, p2: hash(p1) >= hash(p2)
        or reject_functions(p2, "and", "or")
        or reject_functions(p1, "and"),
        "-": lambda p: reject_functions(p, "-", "+", "*", "1/x", "sign", "ite")
        or isinstance(p, Constant),
        "sign": lambda p: reject_functions(p, "-", "1/x", "sign")
        or isinstance(p, Constant),
        "not": lambda p: reject_functions(p, "not", "or", "and"),
    }

    dsl = DSL(primitive_types)
    dsl.instantiate_polymorphic_types(10)
    return (
        dsl,
        DSLEvaluator(dsl.instantiate_semantics(semantics), use_cache=False),
    )
