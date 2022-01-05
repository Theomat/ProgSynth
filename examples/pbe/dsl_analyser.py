from typing import Dict, Generator, List, TypeVar
import copy

import numpy as np

from synth import Dataset, PBE
from synth.pbe import reproduce_dataset
from synth.pruning import UseAllVariablesPruner, SetPruner
from synth.syntax import (
    ConcreteCFG,
    ConcretePCFG,
    enumerate_pcfg,
    DSL,
    Function,
    Primitive,
    Program,
    Variable,
    Arrow,
    Type,
)
from synth.syntax.type_system import FunctionType
from synth.utils import chrono

# ================================
# Change dataset
# ================================
DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"

dataset = DEEPCODER
# ================================
# Tunable parameters
# ================================
max_depth = 2
input_checks = 100
# ================================
# Initialisation
# ================================
dataset_file = f"{dataset}.pickle"
if dataset == DEEPCODER:
    from deepcoder.deepcoder import dsl, evaluator

elif dataset == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, evaluator

# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
print(f"Loading {dataset_file}...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load(dataset_file)
    print("done in", c.elapsed_time(), "s")
# Reproduce dataset distribution
print("Reproducing dataset...", end="")
with chrono.clock("dataset.reproduce") as c:
    task_generator, lexicon = reproduce_dataset(
        full_dataset, dsl, evaluator, 0, uniform_pcfg=True
    )
    print("done in", c.elapsed_time(), "s")
# We only get a task generator for the input generator
input_sampler = task_generator.input_generator

# ================================
# Load dataset & Task Generator
# ================================
simpler_pruner = UseAllVariablesPruner()

sampled_inputs = {}

syntaxic_restrictions = []
specific_restrictions = set()
specific_pruner = SetPruner(specific_restrictions)

stats = {
    s: {"total": 0, "syntaxic": 0}
    for s in ["identity", "symmetry", "invariant", "constant"]
}


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


def vars(program: Program) -> List[Variable]:
    variables = []
    for p in program.depth_first_iter():
        if isinstance(p, Variable):
            variables.append(p)
    return variables


def constants(program: Program) -> List[Variable]:
    consts = []
    for p in program.depth_first_iter():
        if isinstance(p, Primitive) and not isinstance(p.type, Arrow):
            consts.append(p)
    return consts


def add_syntaxic(program: Program):
    pattern = []
    # Find all variables
    for p in program.depth_first_iter():
        if isinstance(p, Primitive):
            pattern.append(p.primitive)
    syntaxic_restrictions.append(pattern)


def program_analysis(program: Program, solutions, category: str):
    prog_vars = vars(program)
    max_vars = len(prog_vars)
    stats[category]["total"] += 1
    # Now more variables have been used
    type_request = FunctionType(*[var.type for var in prog_vars], program.type)
    assert isinstance(type_request, Arrow)
    arguments = type_request.arguments()
    # If only one variable used, this is easy
    if max_vars == 1:
        add_syntaxic(program)
        stats[category]["syntaxic"] += 1
        return

    # Consider all possibilities
    arguments_per_type: Dict[Type, List[Variable]] = {}
    for i, arg_type in enumerate(arguments):
        if arg_type not in arguments_per_type:
            arguments_per_type[arg_type] = []
        arguments_per_type[arg_type].append(Variable(i, arg_type))
    # Enumerate all combinations
    invariant = set()
    identity = set()
    total: int = 0

    # Get inputs
    with chrono.clock("search.input_sampling"):
        if type_request not in sampled_inputs:
            sampled_inputs[type_request] = [
                [input_sampler.sample(type=arg) for arg in arguments]
                for _ in range(input_checks)
            ]
        inputs = sampled_inputs[type_request]

    original = program
    program = copy.deepcopy(original)
    prog_vars = vars(program)
    for i, var in enumerate(prog_vars):
        var.variable = i
    for args in produce_all_variants([arguments_per_type[t] for t in arguments]):
        total += 1
        # Modify program
        for var, arg in zip(prog_vars, args):
            var.variable = arg.variable
        # Init booleans
        is_invariant = original != program
        varno = args[0].variable
        is_identity = all(args[i].variable == varno for i in range(len(arguments)))
        # Check
        for inp, sol in zip(inputs, solutions):
            with chrono.clock("eval"):
                out = evaluator.eval(program, inp)
            if is_invariant and out != sol:
                is_invariant = False
            if is_identity and out != inp[varno]:
                is_identity = False
            if not is_identity and not is_invariant:
                break
        if is_invariant:
            invariant.add(copy.deepcopy(program))
        if is_identity:
            identity.add(copy.deepcopy(program))

    if len(invariant) == total:
        # Then this instead can be thought of as a syntaxic restriction
        add_syntaxic(original)
        stats[category]["syntaxic"] += 1
    else:
        global specific_restrictions
        specific_restrictions |= invariant
        specific_restrictions |= identity


def constant_program_analysis(program: Program):
    category = "constant"
    prog_consts = constants(program)
    max_consts = len(prog_consts)
    stats[category]["total"] += 1

    # If only one constant used, this is easy
    if max_consts == 1:
        add_syntaxic(program)
        stats[category]["syntaxic"] += 1
        return
    else:
        # TODO: this may be done better
        global specific_restrictions
        specific_restrictions.add(program)


with chrono.clock("search"):
    for primitive in dsl.list_primitives:
        arguments = (
            [] if not isinstance(primitive.type, Arrow) else primitive.type.arguments()
        )
        if len(arguments) == 0:
            continue
        # Remove forbidden patterns to speed up search
        dsl.forbidden_patterns = syntaxic_restrictions[:]
        cfg = ConcreteCFG.from_dsl(dsl, primitive.type, max_depth + 1)
        cfg.rules[cfg.start] = {
            P: d for P, d in cfg.rules[cfg.start].items() if P == primitive
        }
        pcfg = ConcretePCFG.uniform(cfg)

        with chrono.clock("search.input_sampling"):
            if primitive.type not in sampled_inputs:
                sampled_inputs[primitive.type] = [
                    [input_sampler.sample(type=arg) for arg in arguments]
                    for _ in range(input_checks)
                ]
            inputs = sampled_inputs[primitive.type]

        with chrono.clock("search.solutions"):
            base_program = Function(
                primitive,
                [Variable(i, arg_type) for i, arg_type in enumerate(arguments)],
            )
            solutions = [evaluator.eval(base_program, inp) for inp in inputs]

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
        with chrono.clock("search.variants.enumeration"):
            for args in produce_all_variants(
                [arguments_per_type[t] for t in arguments]
            ):
                current_prog = Function(
                    primitive,
                    args,
                )
                is_symmetric = current_prog != base_program
                varno = args[0].variable
                is_identity = all(
                    args[i].variable == varno for i in range(len(arguments))
                )
                for inp, sol in zip(inputs, solutions):
                    with chrono.clock("search.variants.enumeration.eval"):
                        out = evaluator.eval(current_prog, inp)
                    if is_symmetric and out != sol:
                        is_symmetric = False
                    if is_identity and out != inp[varno]:
                        is_identity = False
                    if not is_identity and not is_symmetric:
                        break
                if is_symmetric:
                    program_analysis(current_prog, solutions, "symmetry")
                if is_identity:
                    program_analysis(current_prog, solutions, "identity")
        # ========================
        # Invariant part
        # ========================
        with chrono.clock("search.invariant.enumeration"):
            for program in enumerate_pcfg(pcfg, specific_pruner):
                if base_program == program:
                    continue
                with chrono.clock("search.invariant.enumeration.pruning"):
                    if not simpler_pruner.accept((primitive.type, program)):
                        continue
                is_invariant = True
                is_identity = len(arguments) == 1
                for inp, sol in zip(inputs, solutions):
                    with chrono.clock("search.invariant.enumeration.eval"):
                        out = evaluator.eval(program, inp)
                    if is_invariant and out != sol:
                        is_invariant = False
                    if is_identity and out != inp[0]:
                        is_identity = False
                    if not is_identity and not is_invariant:
                        break

                if is_invariant:
                    program_analysis(program, solutions, "invariant")
                if is_identity:
                    program_analysis(program, solutions, "identity")

with chrono.clock("constants"):
    types = set(
        primitive.type
        for primitive in dsl.list_primitives
        if not isinstance(primitive.type, Arrow)
    )
    for ty in types:
        all_evals = {
            evaluator.eval(p, []) for p in dsl.list_primitives if primitive.type == ty
        }
        # Remove forbidden patterns to speed up search
        dsl.forbidden_patterns = syntaxic_restrictions[:]
        cfg = ConcreteCFG.from_dsl(dsl, primitive.type, max_depth + 1)
        pcfg = ConcretePCFG.uniform(cfg)
        with chrono.clock("constants.enumeration"):
            for program in enumerate_pcfg(pcfg, specific_pruner):
                with chrono.clock("constants.enumeration.eval"):
                    out = evaluator.eval(program, [])
                if out in all_evals:
                    constant_program_analysis(program)


print(
    "Done",
    chrono.summary(
        time_formatter=lambda t: f"{int(t*1000)}ms" if not np.isnan(t) else "nan"
    ),
)
print(f"Cache hit rate: {evaluator.cache_hit_rate*100:.1f}%")
print()
print("[=== Report ===]")
for stat_name in stats:
    total = stats[stat_name]["total"]
    if total == 0:
        print(f"Found no {stat_name} simplification")
        continue
    percent = stats[stat_name]["syntaxic"] / total * 100
    print("Found", total, stat_name, f"with {percent:.1f}% syntaxic")

print("[=== Results ===]")
print(f"Found {len(syntaxic_restrictions)} syntaxic restricions.")
copied_dsl = DSL(
    {p.primitive: p.type for p in dsl.list_primitives}, syntaxic_restrictions
)
all_type_requests = set(task_generator.type2pcfg.keys())
dsl.forbidden_patterns = []
cfgs = [ConcreteCFG.from_dsl(dsl, t, 5) for t in all_type_requests]
reduced_cfgs = [ConcreteCFG.from_dsl(copied_dsl, t, 5) for t in all_type_requests]
ratio = np.mean(
    [
        (original.size() - red.size()) / original.size()
        for original, red in zip(cfgs, reduced_cfgs)
    ]
)
mean_ori = np.mean([original.size() for original in cfgs])
mean_red = np.mean([red.size() for red in reduced_cfgs])
print(
    f"This gives an average reduction of CFG size of {ratio * 100:.1f}% from {mean_ori:.0f} to {mean_red:.0f}"
)
print(syntaxic_restrictions)
print()
print(f"Found {len(specific_restrictions)} specific restricions, impact not computed.")
print(specific_restrictions)
