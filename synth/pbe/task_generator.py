from typing import Callable, Dict, Generator, List as TList, Any, Optional, Tuple

import numpy as np

from synth.task import Dataset, Task
from synth.specification import PBE, Example
from synth.semantic.evaluator import Evaluator
from synth.syntax.dsl import DSL
from synth.syntax.program import Program
from synth.syntax.type_system import BOOL, INT, Arrow, List, Type
from synth.syntax.concrete.concrete_cfg import ConcreteCFG
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.generation.sampler import (
    LexiconSampler,
    ListSampler,
    RequestSampler,
    Sampler,
    UnionSampler,
)


class TaskGenerator:
    def __init__(
        self,
        input_generator: Sampler,
        evaluator: Evaluator,
        gen_random_type_request: Sampler[Type],
        gen_random_sample_number: Sampler[int],
        type2pcfg: Dict[Type, ConcretePCFG],
        output_validator: Callable[[Any], bool],
        max_tries: int = 100,
    ) -> None:
        self.input_generator = input_generator
        self.evaluator = evaluator
        self.gen_random_type_request = gen_random_type_request
        self.gen_random_sample_number = gen_random_sample_number
        self.type2pcfg = type2pcfg
        self.max_tries = max_tries
        self.output_validator = output_validator
        # For statistics
        self.difficulty: Dict[Type, TList[int]] = {}

    def __generate_program__(self, type_request: Type) -> Program:
        nargs: int = (
            0 if not isinstance(type_request, Arrow) else len(type_request.arguments())
        )
        solution: Program = self.type2pcfg[type_request].sample_program()
        while not solution.is_using_all_variables(nargs):
            solution = self.type2pcfg[type_request].sample_program()
        return solution

    def generate_task(self) -> Task[PBE]:
        type_request = self.gen_random_type_request.sample()
        arguments = (
            [] if not isinstance(type_request, Arrow) else type_request.arguments()
        )
        if type_request not in self.difficulty:
            self.difficulty[type_request] = [0, 0]

        # Generate correct program that makes use of all variables
        solution = self.__generate_program__(type_request)
        # Try to generate the required number of samples
        samples = self.gen_random_sample_number.sample(type=type_request)
        inputs: TList = []
        outputs = []
        tries = 0
        # has_enough_tries_to_reach_desired_no_of_samples and has_remaining_tries
        while (self.max_tries - tries) + len(
            inputs
        ) >= samples and tries < self.max_tries:
            tries += 1
            new_input = [
                self.input_generator.sample(type=arg_type) for arg_type in arguments
            ]
            try:
                output = self.evaluator.eval(solution, new_input)
            except:
                # Catch errors just in case
                continue
            if self.output_validator(output):
                inputs.append(new_input)
                outputs.append(output)
                if len(inputs) >= samples:
                    break

        self.difficulty[type_request][0] += tries
        self.difficulty[type_request][1] += tries - len(inputs)

        # Sample another task if failed
        if len(inputs) < samples:
            return self.generate_task()
        return Task(
            type_request,
            PBE([Example(inp, out) for inp, out in zip(inputs, outputs)]),
            solution,
            {"generated": True, "tries": tries},
        )

    def generator(self) -> Generator[Task[PBE], None, None]:
        while True:
            yield self.generate_task()


def basic_output_validator(
    int_lexicon: TList[int], max_list_length: int
) -> Callable[[Any], bool]:
    def validate_output(output: Any) -> bool:
        if isinstance(output, bool):
            return True
        elif isinstance(output, int):
            return output in int_lexicon
        elif isinstance(output, list):
            return len(output) <= max_list_length and all(
                validate_output(x) for x in output
            )
        return False

    return validate_output


def reproduce_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    max_tries: int = 100,
    int_bound: int = 1000
) -> Tuple[TaskGenerator, TList[int]]:

    max_depth = -1
    allowed_types: TList[Type] = []
    types_probs_list: TList[float] = []

    list_length: Dict[Type, Dict[int, int]] = {}
    no_samples: Dict[Type, Dict[int, int]] = {}
    max_list_depth = [0]

    int_range: TList[int] = [999999999, 0]
    int_range[1] = -int_range[0]

    def analyze(element: Any, type: Type, depth: int = 1) -> None:
        if depth > max_list_depth[0]:
            max_list_depth[0] = depth
        if isinstance(type, List):
            elt_type = type.element_type
            if len(element) > 0:
                __multi_discrete_distribution__(list_length, elt_type, len(element))
                for el in element:
                    analyze(el, elt_type, depth + 1)
        elif element:
            int_range[0] = min(int_range[0], max(-int_bound, element))
            int_range[1] = max(int_range[1], min(int_bound, element))

    # Capture all information in one dataset pass
    for task in dataset:
        if task.solution:
            max_depth = max(max_depth, task.solution.depth())
        # Type distribution
        if task.type_request not in allowed_types:
            allowed_types.append(task.type_request)
            types_probs_list.append(1.0)
        else:
            index = allowed_types.index(task.type_request)
            types_probs_list[index] += 1.0

        # No samples distribution
        __multi_discrete_distribution__(
            no_samples, task.type_request, len(task.specification.examples)
        )

        t = task.type_request
        args = [] if not isinstance(t, Arrow) else t.arguments()
        r = t if not isinstance(t, Arrow) else t.returns()

        # Input data analysis
        for ex in task.specification.examples:
            for input, ti in zip(ex.inputs, args):
                analyze(input, ti)
            analyze(ex.output, r)

    # Type generator
    types_probs = np.array(types_probs_list, dtype=float) / len(dataset)
    type_sampler = LexiconSampler(allowed_types, types_probs, seed)

    list_length_gen = __multi_discrete_to_gen__(list_length, seed=seed)
    no_samples_gen = __multi_discrete_to_gen__(no_samples, seed=seed)

    int_lexicon = list(range(int_range[0], int_range[1] + 1))

    type2PCFG = {
        t: ConcretePCFG.uniform_from_cfg(ConcreteCFG.from_dsl(dsl, t, max_depth))
        for t in allowed_types
    }
    for pcfg in type2PCFG.values():
        pcfg.init_sampling(seed)

    input_sampler = ListSampler(
        UnionSampler(
            {
                INT: LexiconSampler(int_lexicon, seed=seed),
                BOOL: LexiconSampler([True, False], seed=seed),
            }
        ),
        list_length_gen,
        max_depth=max_list_depth[0],
        seed=seed,
    )

    return (
        TaskGenerator(
            input_sampler,
            evaluator,
            type_sampler,
            no_samples_gen,
            type2PCFG,
            basic_output_validator(
                int_lexicon, max(max(l.keys()) for l in list_length.values())
            ),
            max_tries,
        ),
        int_lexicon,
    )


def __multi_discrete_distribution__(
    distr: Dict[Type, Dict[int, int]], key: Type, new_entry: int
) -> None:
    if key not in distr:
        distr[key] = {new_entry: 0}
    elif new_entry not in distr[key]:
        distr[key][new_entry] = 0
    distr[key][new_entry] += 1


def __multi_discrete_to_gen__(
    distr: Dict[Type, Dict[int, int]], seed: Optional[int] = None
) -> RequestSampler[int]:

    choice_map: Dict[Type, TList[int]] = {k: list(v.keys()) for k, v in distr.items()}
    probs_map: Dict[Type, np.ndarray] = {
        k: np.array(list(v.values()), dtype=float) for k, v in distr.items()
    }
    for k, v in probs_map.items():
        probs_map[k] /= np.sum(v)

    samplers: Dict[Type, Sampler] = {
        k: LexiconSampler(choice_map[k], probs_map[k], seed=seed)
        for k in probs_map.keys()
    }
    return UnionSampler(samplers)
