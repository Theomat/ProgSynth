from typing import Any, Callable, Optional, Tuple, List as TList, Union

import numpy as np

from synth.generation.sampler import LexiconSampler, UnionSampler
from synth.pbe.task_generator import (
    TaskGenerator,
    basic_output_validator,
    reproduce_dataset,
)
from synth.semantic import DSLEvaluator, Evaluator
from synth.specification import PBE
from synth.syntax import (
    DSL,
    INT,
    Arrow,
    FixedPolymorphicType,
    PrimitiveType,
    BOOL,
    FunctionType,
    auto_type,
)
from synth.task import Dataset

# a type representing either an int or a float
FLOAT = PrimitiveType("float")
type = FixedPolymorphicType("int/float", INT | FLOAT)

__semantics = {
    "+": lambda a: lambda b: round(a + b, 1),
    "-": lambda a: lambda b: round(a - b, 1),
    "int2float": lambda a: float(a),
    "1": 1,
    "2": 2,
    "3": 3,
    "3.0": 3.0,
}

__primitive_types = {
    # int|float -> int|float -> int|float
    "+": Arrow(type, Arrow(type, type)),
    "-": FunctionType(type, type, type),
    "int2float": auto_type("int->float"),
    "1": INT,
    "2": INT,
    "3": INT,
}

# Short example of a forbidden patterns (if add1 and sub1 are defined in _semantics and _primitive_types)
_forbidden_patterns = {
    ("add1", 0): {"sub1"},
    ("sub1", 0): {"add1"},
}

dsl = DSL(__primitive_types, forbidden_patterns=_forbidden_patterns)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
lexicon = [round(x, 1) for x in np.arange(-256, 256 + 1, 0.1)]


def reproduce_calculator_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    int_bound: int = 1000,
    *args: Any,
    **kwargs: Any
) -> Tuple[TaskGenerator, TList[int]]:

    int_range: TList[int] = [int_bound, 0]
    int_range[1] = -int_range[0]

    float_range: TList[float] = [float(int_bound), 0]
    float_range[1] = -float_range[0]
    float_bound = float(int_bound)

    def analyser(start: None, element: Union[int, float]) -> None:
        if isinstance(element, int):
            int_range[0] = min(int_range[0], max(-int_bound, element))
            int_range[1] = max(int_range[1], min(int_bound, element))
        elif isinstance(element, float):
            float_range[0] = min(float_range[0], max(-float_bound, element))
            float_range[1] = max(float_range[1], min(float_bound, element))

    def get_element_sampler(start: None) -> UnionSampler:
        int_lexicon = list(range(int_range[0], int_range[1] + 1))
        float_lexicon = [
            round(x, 1) for x in np.arange(float_range[0], float_range[1] + 1, 0.1)
        ]
        return UnionSampler(
            {
                INT: LexiconSampler(int_lexicon, seed=seed),
                BOOL: LexiconSampler([True, False], seed=seed),
                FLOAT: LexiconSampler(float_lexicon, seed=seed),
            }
        )

    def get_validator(start: None, max_list_length: int) -> Callable[[Any], bool]:
        return basic_output_validator(
            {
                int: list(range(int_range[0], int_range[1] + 1)),
                float: [
                    round(x, 1)
                    for x in np.arange(float_range[0], float_range[1] + 1, 0.1)
                ],
            },
            max_list_length,
        )

    def get_lexicon(start: None) -> TList[float]:
        return [round(x, 1) for x in np.arange(float_range[0], float_range[1] + 1, 0.1)]

    return reproduce_dataset(
        dataset,
        dsl,
        evaluator,
        None,
        analyser,
        get_element_sampler,
        get_validator,
        get_lexicon,
        seed,
        *args,
        **kwargs
    )
