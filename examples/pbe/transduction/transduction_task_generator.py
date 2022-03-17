from random import randint
import re
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List as TList,
    Any,
    Optional,
    Set,
    Tuple,
    Type as PythonType,
)

import numpy as np

from synth.task import Dataset, Task
from synth.specification import PBE, Example
from synth.semantic.evaluator import Evaluator
from synth.syntax.dsl import DSL
from synth.syntax.program import Program
from synth.syntax.type_system import Arrow, List, Type, STRING, PrimitiveType
from synth.syntax.concrete.concrete_cfg import ConcreteCFG
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.generation.sampler import (
    LexiconSampler,
    ListSampler,
    RequestSampler,
    Sampler,
    UnionSampler,
)
from examples.pbe.regexp.type_regex import REGEXP

class TaskGenerator:
    def __init__(
        self,
        input_generator: Sampler,
        evaluator: Evaluator,
        gen_random_type_request: Sampler[Type],
        gen_random_sample_number: Sampler[str],
        pcfgs: Iterable[ConcretePCFG],
        output_validator: Callable[[Any], bool],
        max_tries: int = 100,
        skip_exceptions: Optional[Set[PythonType]] = None,
    ) -> None:
        self.input_generator = input_generator
        self.evaluator = evaluator
        self.gen_random_type_request = gen_random_type_request
        self.gen_random_sample_number = gen_random_sample_number
        self.type2pcfg = {pcfg.type_request: pcfg for pcfg in pcfgs}
        self.max_tries = max_tries
        self.output_validator = output_validator
        self.skip_exceptions = skip_exceptions or set()

        self._failed_types: Set[Type] = set()
        # For statistics
        self.difficulty: Dict[Type, TList[int]] = {}
        self.generated_types: Dict[Type, int] = {t: 0 for t in self.type2pcfg}

    def __generate_program__(self, type_request: Type) -> Program:
        nargs: int = (
            0 if not isinstance(type_request, Arrow) else len(type_request.arguments())
        )
        solution: Program = self.type2pcfg[type_request].sample_program()
        tries: int = 0
        var_used = len(solution.used_variables())
        best = solution
        while var_used < nargs and tries < self.max_tries:
            solution = self.type2pcfg[type_request].sample_program()
            tries += 1
            n = len(solution.used_variables())
            if n > var_used:
                var_used = n
                best = solution
        return best

    def generate_task(self) -> Task[PBE]:
        type_request = self.gen_random_type_request.sample()
        i = 0
        while type_request in self._failed_types and i <= self.max_tries:
            type_request = self.gen_random_type_request.sample()
            i += 1
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
            except Exception as e:
                if type(e) in self.skip_exceptions:
                    continue
                else:
                    raise e
            if self.output_validator(output):
                # print(f"input = {new_input}, output = {output}, solution = {solution}")
                inputs.append(new_input)
                outputs.append(output)
                if len(inputs) >= samples:
                    break

        self.difficulty[type_request][0] += tries
        self.difficulty[type_request][1] += tries - len(inputs)

        # Sample another task if failed
        if len(inputs) < samples:
            self._failed_types.add(type_request)
            return self.generate_task()
        self._failed_types = set()
        self.generated_types[type_request] += 1
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
    str_lexicon: TList[str], max_list_length: int
) -> Callable[[Any], bool]:
    def validate_output(output: Any) -> bool:
        if isinstance(output, str):
            return output in str_lexicon
        elif isinstance(output, list):
            return (max_list_length < 0 or len(output) <= max_list_length) and all(
                validate_output(x) for x in output
            )
        return False

    return validate_output


def reproduce_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    uniform_pcfg: bool = True,
    max_tries: int = 100,
    default_max_depth: int = 10,
    max_list_length: Optional[int] = None,
) -> Tuple[TaskGenerator, TList[int]]:

    max_depth = -1
    allowed_types: TList[Type] = []
    types_probs_list: TList[float] = []

    list_length: Dict[Type, Dict[int, int]] = {}
    no_samples: Dict[Type, Dict[int, int]] = {}
    max_list_depth = [0]

    def analyze(element: Any, type: Type, depth: int = 1) -> None:
        if depth > max_list_depth[0]:
            max_list_depth[0] = depth
        if isinstance(type, List):
            elt_type = type.element_type
            if len(element) > 0:
                __multi_discrete_distribution__(list_length, type, len(element))
                for el in element:
                    analyze(el, elt_type, depth + 1)

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

    list_length_gen = __multi_discrete_to_gen__(
        list_length, seed=seed, maxi=max_list_length
    )
    no_samples_gen = __multi_discrete_to_gen__(no_samples, seed=seed)

    if max_depth == -1:
        max_depth = default_max_depth
    if uniform_pcfg:
        pcfgs = {
            ConcretePCFG.uniform(ConcreteCFG.from_dsl(dsl, t, max_depth))
            for t in allowed_types
        }
    else:
        pcfgs = {
            dataset.to_pcfg(ConcreteCFG.from_dsl(dsl, t, max_depth), filter=True)
            for t in allowed_types
        }
    for pcfg in pcfgs:
        pcfg.init_sampling(seed)

    str_lexicon = list([chr(i) for i in range(32, 126)])
    input_sampler = ListSampler(
        UnionSampler(
            {
                STRING: LexiconSampler(str_lexicon, seed=seed),
                REGEXP: LexiconSampler(['_', ')', '{', '+', ';', '=', '$', '\\', '^', ',', '!', '*', "'", ' ', '>', '}', '<', '[', '"', '#', '|', '`', '%', '?', ':', ']', '&', '(', '@', '.', '/', '-'], seed=seed),
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
            pcfgs,
            basic_output_validator(
                str_lexicon,
                max_list_length
                or max(
                    (max(l.keys()) for l in list_length.values()), default=-1
                ),  # type:ignore
            ),
            max_tries,
        ),
        str_lexicon,
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
    distr: Dict[Type, Dict[int, int]],
    seed: Optional[int] = None,
    maxi: Optional[int] = None,
) -> RequestSampler[int]:

    choice_map: Dict[Type, TList[int]] = {k: list(v.keys()) for k, v in distr.items()}
    probs_map: Dict[Type, np.ndarray] = {
        k: np.array(list(v.values()), dtype=float) for k, v in distr.items()
    }
    if maxi:
        for k, v in distr.items():
            changed = False
            added = 0
            for length, qty in v.items():
                if length > maxi:
                    added = qty
                    choice_map[k].remove(length)
                    changed = True
            if changed:
                if maxi not in choice_map[k]:
                    choice_map[k].append(maxi)
                probs_map[k] = np.array(
                    [distr[k].get(z, added) for z in choice_map[k]], dtype=float
                )
    for k, val in probs_map.items():
        probs_map[k] /= np.sum(val)

    samplers: Dict[Type, Sampler] = {
        k: LexiconSampler(choice_map[k], probs_map[k], seed=seed)
        for k in probs_map.keys()
    }
    return UnionSampler(samplers)
