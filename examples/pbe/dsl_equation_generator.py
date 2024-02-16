from collections import defaultdict
import json
from typing import Any, Callable, Dict, Generator, List, Set, Tuple, TypeVar
import argparse

import tqdm
import numpy as np
from colorama import Fore as F

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from equivalence_classes_to_filter import equivalence_classes_to_filters


from synth import Dataset, PBE
from synth.generation.sampler import Sampler
from synth.pbe import reproduce_dataset
from synth.filter import Filter
from synth.specification import PBEWithConstants
from synth.syntax import (
    CFG,
    DetGrammar,
    ProbDetGrammar,
    bps_enumerate_prob_grammar as enumerate_prob_grammar,
    Function,
    Primitive,
    Program,
    Variable,
    Type,
    ProgramEnumerator,
)
from synth.utils import chrono


parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "--no-reproduce",
    action="store_true",
    default=False,
    help="instead of trying to sample new inputs, scrap inputs from the dataset",
)
parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
parser.add_argument(
    "--n", type=int, default=500, help="number of examples to be sampled (default: 500)"
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=2,
    help="max depth of programs to check for (default: 2)",
)


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
input_checks: int = parameters.n
max_depth: int = parameters.max_depth
no_reproduce: bool = parameters.no_reproduce
seed: int = parameters.seed
# ================================
# Initialisation
# ================================
dsl_module = load_DSL(dsl_name)
dsl, evaluator = dsl_module.dsl, dsl_module.evaluator

# Load dataset
full_dataset: Dataset[PBE] = load_dataset(dsl_name, dataset_file)

our_eval = lambda *args: evaluator.eval(*args)


# ================================
# Produce data samplers
# ================================
if not no_reproduce:
    if hasattr(dsl_module, "reproduce_dataset"):
        reproduce_dataset = dsl_module.reproduce_dataset
    else:
        from synth.pbe.task_generator import reproduce_int_dataset as reproduce_dataset

    # Reproduce dataset distribution
    print("Reproducing dataset...", end="", flush=True)
    with chrono.clock("dataset.reproduce") as c:
        task_generator, _ = reproduce_dataset(
            full_dataset,
            dsl,
            evaluator,
            0,
            default_max_depth=max_depth,
            uniform_pgrammar=True,
        )
        print("done in", c.elapsed_time(), "s")
    # We only get a task generator for the input generator
    input_sampler = task_generator.input_generator
else:
    inputs_from_type = defaultdict(list)
    for task in full_dataset:
        for ex in task.specification.examples:
            for inp, arg in zip(ex.inputs, task.type_request.arguments()):
                inputs_from_type[arg].append(inp)
        # Manage input/output constants
        if isinstance(task.specification, PBEWithConstants):
            for ct, values in task.specification.constants:
                inputs_from_type[ct] += values

    class SamplesSampler(Sampler):
        def __init__(self, samples: Dict[Type, List[Any]], seed: int) -> None:
            self._gen = np.random.default_rng(seed=seed)
            self.samples = samples

        def sample(self, type: Type, **kwargs: Any) -> Any:
            return self._gen.choice(self.samples[type])

    input_sampler = SamplesSampler(inputs_from_type, seed=seed)


# =========================================================================================
# Equivalence classes
# =========================================================================================

equivalence_classes = defaultdict(dict)
commutatives = []
constants = []
identities = []
n_equiv_classes = 0


def new_equivalence_class(program: Program) -> None:
    global n_equiv_classes
    equivalence_classes[program] = n_equiv_classes
    n_equiv_classes += 1


def merge_equivalence_classes(program: Program, representative: Program) -> None:
    equivalence_classes[program] = equivalence_classes[representative]


def get_equivalence_class(num: int) -> Set[Program]:
    return {p for p, v in equivalence_classes.items() if v == num}


# =========================================================================================
# UTILS
# =========================================================================================

T = TypeVar("T")


def produce_all_variants(possibles: List[List[T]]) -> Generator[List[T], None, None]:
    # Enumerate all combinations
    n = len(possibles)
    maxi = [len(possibles[i]) for i in range(n)]
    current = [0 for _ in range(n)]
    while current[0] < maxi[0]:
        yield [possibles[i][j] for i, j in enumerate(current)]
        # Next combination
        i = n - 1
        current[i] += 1
        while i > 0 and current[i] >= maxi[i]:
            current[i] = 0
            i -= 1
            current[i] += 1


# =========================================================================================
# =========================================================================================

sampled_inputs = {}
all_solutions: Dict[Type, Dict[Program, List]] = defaultdict(dict)
programs_done = set()
forbidden_types = set()


def init_base_primitives() -> None:
    """
    Init sampled data types and create signatures for all base primitives
    """
    primitives: List[Primitive] = dsl.list_primitives
    # Check forbidden types
    all_types = set()
    for primitive in primitives:
        all_types |= primitive.type.decompose_type()[0]
    for arg_type in all_types:
        try:
            input_sampler.sample(type=arg_type)
        except:
            forbidden_types.add(arg_type)
    # Pre Sample Inputs + Pre Execute base primitives
    for primitive in primitives:
        arguments = primitive.type.arguments()
        if len(arguments) == 0 or any(arg in forbidden_types for arg in arguments):
            continue
        if primitive.type not in sampled_inputs:
            sampled_inputs[primitive.type] = [
                [input_sampler.sample(type=arg) for arg in arguments]
                for _ in range(input_checks)
            ]
        inputs = sampled_inputs[primitive.type]
        base_program = Function(
            primitive,
            [Variable(i, arg_type) for i, arg_type in enumerate(arguments)],
        )
        programs_done.add(base_program)
        solutions = [our_eval(base_program, inp) for inp in inputs]
        all_solutions[base_program.type.returns()][base_program] = solutions
        new_equivalence_class(base_program)


def check_program(
    program: Program, inputs: List, all_sol: Dict[Program, List]
) -> Tuple[bool, bool, Set[Program], List[Any]]:
    is_constant = True
    is_list_constant = True
    my_outputs = []
    candidates = set(all_sol.keys())
    is_identity = [len(program.used_variables()) == 1 for _ in program.type.arguments()]
    for i, inp in enumerate(inputs):
        out = our_eval(program, inp)
        # Update candidates
        candidates = {c for c in candidates if all_sol[c][i] == out}
        is_identity = [x and out == inp[i] for i, x in enumerate(is_identity)]
        if is_constant and len(my_outputs) > 0 and my_outputs[-1] != out:
            is_constant = False
        is_list_constant = (
            is_list_constant
            and isinstance(out, List)
            and all(x == out[0] for x in out)  # list with only one same element
            and (
                len(my_outputs) == 0  # no input so far
                or len(out) == 0  # out is empty list
                or all(
                    len(x) == 0 for x in my_outputs
                )  # all elements previously in my_outputs are empty
                or [x for x in my_outputs if len(x) > 0][0][0]
                == out[0]  # sole value of out == previous sole value of my_outputs
            )
        )
        my_outputs.append(out)
    return is_constant or is_list_constant, any(is_identity), candidates, my_outputs


def check_symmetries() -> None:
    """
    Try to find symmetries (commutativity)
    """
    iterable = tqdm.tqdm(dsl.list_primitives)
    for primitive in iterable:
        arguments = primitive.type.arguments()
        if len(arguments) == 0 or any(arg in forbidden_types for arg in arguments):
            continue
        iterable.set_postfix_str(primitive.primitive)

        base_program = Function(
            primitive,
            [Variable(i, arg_type) for i, arg_type in enumerate(arguments)],
        )
        inputs = sampled_inputs[primitive.type]
        all_sol = all_solutions[base_program.type.returns()]

        # ========================
        # Symmetry+Identity part
        # ========================
        # Fill arguments per type
        arguments_per_type: Dict[Type, List[Variable]] = {}
        for i, arg_type in enumerate(arguments):
            if arg_type not in arguments_per_type:
                arguments_per_type[arg_type] = []
            arguments_per_type[arg_type].append(Variable(i, arg_type))
        # Enumerate all combinations
        for args in produce_all_variants([arguments_per_type[t] for t in arguments]):
            current_prog = Function(
                primitive,
                args,
            )
            if current_prog in programs_done:
                continue
            programs_done.add(current_prog)
            is_constant, is_identity, candidates, my_outputs = check_program(
                current_prog, inputs, all_sol
            )
            is_symmetric = base_program in candidates
            if is_identity:
                identities.append(current_prog)
            elif is_constant:
                constants.append(current_prog)
            elif is_symmetric:
                commutatives.append(current_prog)
                merge_equivalence_classes(current_prog, base_program)
            else:
                new_equivalence_class(base_program)
                all_sol[current_prog] = my_outputs


def check_equivalent() -> None:
    ftypes = tqdm.tqdm(sampled_inputs.keys())
    for ftype in ftypes:
        cfg = CFG.depth_constraint(dsl, ftype, max_depth + 1)

        inputs = sampled_inputs[ftype]
        all_sol = all_solutions[ftype.returns()]

        cfg_size = cfg.programs()
        ftypes.set_postfix_str(f"{F.GREEN}{0 / cfg_size:.0%}{F.RESET}")

        # ========================
        # Check all programs starting with max depth
        # ========================
        for done, program in enumerate(get_enumerator(cfg)):
            if program in programs_done:
                continue
            is_constant, is_identity, candidates, my_outputs = check_program(
                program, inputs, all_sol
            )
            if is_identity:
                identities.append(program)
            elif is_constant:
                constants.append(program)
            elif len(candidates) > 0:
                merge_equivalence_classes(program, list(candidates)[0])
            else:
                new_equivalence_class(program)
                all_sol[program] = my_outputs
            ftypes.set_postfix_str(f"{F.GREEN}{done / cfg_size:.0%}{F.RESET}")
    ftypes.close()


def get_equivalence_classes() -> List[Set[Program]]:
    classes = [get_equivalence_class(i) for i in range(n_equiv_classes)]
    classes.append(identities + constants + [Variable(0)])
    classes = [l for l in classes if len(l) > 1]
    return classes


def update_filter(
    verbose: bool = True,
) -> Tuple[Callable[[Type], Filter[Program]], Dict[str, float]]:
    classes = get_equivalence_classes()
    if len(classes) == 0:
        return lambda x: None, {}
    if verbose:
        print(
            f"\tcurrently found {F.YELLOW}{len(classes)}{F.RESET} equivalence classes"
        )
    builder = equivalence_classes_to_filters(commutatives, classes, dsl)
    if verbose:
        print(
            f"\tfound {F.YELLOW}{builder.stats['constraints.successes']}{F.RESET} ({F.YELLOW}{builder.stats['constraints.successes']/builder.stats['constraints.total']:.1%}{F.RESET}) constraints"
        )
    return (lambda t: builder.get_filter(t, set())), builder.stats


def get_enumerator(cfg: DetGrammar) -> ProgramEnumerator:
    pcfg = ProbDetGrammar.uniform(cfg)
    enumerator = enumerate_prob_grammar(pcfg)
    enumerator.filter = update_filter(False)[0](pcfg.type_request)
    return enumerator


def reduced_explosion() -> Tuple[float, float]:
    stats = update_filter(False)[1]
    ratio_added = stats["constraints.successes"] / stats["constraints.total"]
    ratio_size = stats["dfta.size.final"] / stats["dfta.size.initial"]
    return ratio_added, ratio_size


init_base_primitives()
check_symmetries()
check_equivalent()

print(f"Cache hit rate: {evaluator.cache_hit_rate:.1%}")
print()

classes = get_equivalence_classes()
with open(f"equivalent_classes_{dsl_name}.json", "w") as fd:
    my_list = [list(map(str, l)) for l in classes]
    json.dump(
        {
            "classes": my_list,
            "commutatives": list(map(str, commutatives)),
            "identities": list(map(str, identities)),
            "constants": list(map(str, constants)),
        },
        fd,
    )


print(f"Data saved to {F.GREEN}equivalent_classes_{dsl_name}.json{F.RESET}.")

print(f"Found {F.GREEN}{len(classes)}{F.RESET} equivalence classes.")
print(
    f"Found {F.GREEN}{len(identities)}{F.RESET} programs that were the identify function."
)
print(f"Found {F.GREEN}{len(constants)}{F.RESET} programs that were a constant.")
print(f"Found {F.GREEN}{len(commutatives)}{F.RESET} instances of commutativity.")
print()
r1, r2 = reduced_explosion()
print(f"converted {F.GREEN}{r1:.1%}{F.RESET} of constraints found.")
print(f"reduced combinatorial explosion to {F.GREEN}{r2:.1%}{F.RESET} of original.")
