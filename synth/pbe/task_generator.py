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
    TypeVar,
    Union,
)
from collections.abc import Container

import numpy as np

from synth.filter.constraints.dfta_constraints import add_dfta_constraints
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.grammars.u_cfg import UCFG
from synth.task import Dataset, Task
from synth.specification import PBE, Example
from synth.semantic.evaluator import Evaluator
from synth.syntax.dsl import DSL
from synth.syntax.program import Program
from synth.syntax.type_system import BOOL, INT, List, Type
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
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
        pgrammars: Iterable[Union[ProbUGrammar, ProbDetGrammar]],
        output_validator: Callable[[Any], bool],
        max_tries: int = 100,
        uniques: bool = False,
        skip_exceptions: Optional[Set[PythonType]] = None,
        verbose: bool = False,
    ) -> None:
        self.input_generator = input_generator
        self.evaluator = evaluator
        self.gen_random_type_request = gen_random_type_request
        self.gen_random_sample_number = gen_random_sample_number
        self.type2pgrammar = {pgrammar.type_request: pgrammar for pgrammar in pgrammars}
        self.max_tries = max_tries
        self.output_validator = output_validator
        self.skip_exceptions = skip_exceptions or set()
        self.uniques = uniques
        self.seen: Set[Program] = set()
        self.verbose = verbose

        self._failed_types: Set[Type] = set()
        # For statistics
        self.difficulty: Dict[Type, TList[int]] = {
            t: [0, 0] for t in self.type2pgrammar
        }
        self.generated_types: Dict[Type, int] = {t: 0 for t in self.type2pgrammar}

    def generate_program(self, type_request: Type) -> Tuple[Program, bool]:
        """
        Returns (program, is_unique)
        """
        nargs: int = len(type_request.arguments())
        solution: Program = self.type2pgrammar[type_request].sample_program()
        tries: int = 0
        unique_tries: int = 0
        while solution in self.seen and unique_tries < self.max_tries:
            solution = self.type2pgrammar[type_request].sample_program()
            unique_tries += 1

        var_used = len(solution.used_variables())
        best = solution
        tries = 0
        while var_used < nargs and tries < self.max_tries:
            solution = self.type2pgrammar[type_request].sample_program()
            while solution in self.seen and unique_tries < self.max_tries:
                solution = self.type2pgrammar[type_request].sample_program()
                unique_tries += 1
            tries += 1
            n = len(solution.used_variables())
            if n > var_used:
                var_used = n
                best = solution
        return best, unique_tries < self.max_tries

    def generate_type_request(self) -> Type:
        type_request = self.gen_random_type_request.sample()
        i = 0
        while type_request in self._failed_types and i <= self.max_tries:
            type_request = self.gen_random_type_request.sample()
            i += 1
        if type_request not in self.difficulty:
            self.difficulty[type_request] = [0, 0]
        return type_request

    def sample_input(self, arguments: TList[Type]) -> TList:
        return [self.input_generator.sample(type=arg_type) for arg_type in arguments]

    def eval_input(self, solution: Program, input: TList) -> Any:
        try:
            return self.evaluator.eval(solution, input)
        except Exception as e:
            if type(e) in self.skip_exceptions:
                return None
            else:
                raise e

    def make_task(
        self,
        type_request: Type,
        solution: Program,
        inputs: TList,
        outputs: TList,
        **kwargs: Any
    ) -> Task[PBE]:
        return Task(
            type_request,
            PBE([Example(inp, out) for inp, out in zip(inputs, outputs)]),
            solution,
            {"generated": True, **kwargs},
        )

    def generate_task(self) -> Task[PBE]:
        self._failed_types.clear()
        while True:
            type_request = self.generate_type_request()
            arguments = type_request.arguments()

            # Generate correct program that makes use of all variables
            solution, is_unique = self.generate_program(type_request)
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
                new_input = self.sample_input(arguments)
                output = self.eval_input(solution, new_input)

                if self.output_validator(output) and output not in outputs:
                    inputs.append(new_input)
                    outputs.append(output)
                    if len(inputs) >= samples:
                        break

            self.difficulty[type_request][0] += tries
            self.difficulty[type_request][1] += tries - len(inputs)

            # Sample another task if failed
            if len(inputs) < samples:
                self._failed_types.add(type_request)
                continue
            self._failed_types = set()
            self.generated_types[type_request] += 1
            if self.uniques and is_unique:
                self.seen.add(solution)
            elif self.verbose and not self.uniques:
                print(
                    "Generated a copy of an existing program for type request:",
                    type_request,
                    "program:",
                    solution,
                )
            return self.make_task(
                type_request,
                solution,
                inputs,
                outputs,
                tries=tries,
                unique=is_unique,
            )

    def generator(self) -> Generator[Task[PBE], None, None]:
        while True:
            yield self.generate_task()


def basic_output_validator(
    dico: Dict[PythonType, Container], max_list_length: int
) -> Callable[[Any], bool]:
    def validate_output(output: Any) -> bool:
        if isinstance(output, list):
            return (max_list_length < 0 or len(output) <= max_list_length) and all(
                validate_output(x) for x in output
            )
        else:
            return output in dico.get(type(output), [])
        return False

    return validate_output


def reproduce_int_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    uniform_pgrammar: bool = True,
    max_tries: int = 100,
    int_bound: int = 1000,
    default_max_depth: int = 5,
    max_list_length: Optional[int] = None,
    **kwargs: Any
) -> Tuple[TaskGenerator, TList[int]]:

    int_range: TList[int] = [999999999, 0]
    int_range[1] = -int_range[0]

    def analyser(start: None, element: int) -> None:
        int_range[0] = min(int_range[0], max(-int_bound, element))
        int_range[1] = max(int_range[1], min(int_bound, element))

    def get_element_sampler(start: None) -> UnionSampler:
        int_lexicon = list(range(int_range[0], int_range[1] + 1))

        return UnionSampler(
            {
                INT: LexiconSampler(int_lexicon, seed=seed),
                BOOL: LexiconSampler([True, False], seed=seed),
            }
        )

    def get_validator(start: None, max_list_length: int) -> Callable[[Any], bool]:
        int_lexicon = list(range(int_range[0], int_range[1] + 1))
        return basic_output_validator(
            {int: int_lexicon, bool: {True, False}}, max_list_length
        )

    def get_lexicon(start: None) -> TList[int]:
        return list(range(int_range[0], int_range[1] + 1))

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
        uniform_pgrammar,
        max_tries,
        default_max_depth,
        max_list_length,
        **kwargs
    )


T = TypeVar("T")


def reproduce_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    start: T,
    element_analyser: Callable[[T, Any], T],
    get_element_sampler: Callable[[T], Sampler],
    get_validator: Callable[[T, int], Callable[[Any], bool]],
    get_lexicon: Callable[[T], TList],
    seed: Optional[int] = None,
    uniform_pgrammar: bool = True,
    max_tries: int = 100,
    default_max_depth: int = 5,
    max_list_length: Optional[int] = None,
    constraints: TList[str] = [],
    **kwargs: Any
) -> Tuple[TaskGenerator, TList]:
    """

    start = element_analyser(start, element)
        called when encountering a base element (not a list)
    get_element_sampler(start)
        produces the sampler used for base types (not list)
    get_validator(start, max_list_length)
        produces the output validator
    get_lexicon(start)
        produces the lexicon
    """

    max_depth = -1
    allowed_types: TList[Type] = []
    types_probs_list: TList[float] = []

    list_length: Dict[Type, Dict[int, int]] = {}
    no_samples: Dict[Type, Dict[int, int]] = {}
    max_list_depth = [0]

    out = [start]

    def analyze(element: Any, type: Type, depth: int = 1) -> None:
        if depth > max_list_depth[0]:
            max_list_depth[0] = depth
        if type.is_instance(List):
            elt_type: Type = type.types[0]  # type: ignore
            if len(element) > 0:
                __multi_discrete_distribution__(list_length, type, len(element))
                for el in element:
                    analyze(el, elt_type, depth + 1)
        elif element:
            out[0] = element_analyser(out[0], element)

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
        args = t.arguments()
        r = t.returns()

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
    if uniform_pgrammar:
        pgrammars: Set[Union[ProbUGrammar, ProbDetGrammar]] = set()
        if constraints:
            pgrammars = {
                ProbUGrammar.uniform(
                    UCFG.from_DFTA_with_ngrams(
                        add_dfta_constraints(
                            CFG.depth_constraint(dsl, t, max_depth),
                            constraints,
                            progress=False,
                        ),
                        2,
                    )
                )
                for t in allowed_types
            }
        else:
            pgrammars = {
                ProbDetGrammar.uniform(CFG.depth_constraint(dsl, t, max_depth))
                for t in allowed_types
            }
    else:
        type2grammar = {
            t: CFG.depth_constraint(dsl, t, max_depth) for t in allowed_types
        }
        type2samples = {
            t: [
                task.solution
                for task in dataset
                if task.solution and (t == task.type_request)
            ]
            for t in allowed_types
        }
        pgrammars = {
            ProbDetGrammar.pcfg_from_samples(
                type2grammar[t], [sol for sol in type2samples[t] if sol]
            )
            for t in allowed_types
        }
    for pgrammar in pgrammars:
        pgrammar.init_sampling(seed)

    input_sampler = ListSampler(
        get_element_sampler(out[0]),
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
            pgrammars,
            get_validator(
                out[0],
                max_list_length
                or max((max(l.keys()) for l in list_length.values()), default=-1)
                or -1,
            ),
            max_tries,
            **kwargs
        ),
        get_lexicon(out[0]),
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
