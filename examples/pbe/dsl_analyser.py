from collections import defaultdict
import itertools
from typing import Dict, Generator, List, Set, Tuple, TypeVar
import copy

import tqdm
import numpy as np

from synth import Dataset, PBE
from synth.pbe import reproduce_dataset
from synth.pruning import UseAllVariablesPruner
from synth.syntax import (
    CFG,
    ProbDetGrammar,
    enumerate_prob_grammar,
    DSL,
    Function,
    Primitive,
    Program,
    Variable,
    Arrow,
    Type,
)
from synth.tools.type_constraints.utils import SYMBOL_ANYTHING, SYMBOL_FORBIDDEN
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
input_checks = 500
in_depth_equivalence_check = True
progress = True
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
        full_dataset, dsl, evaluator, 0, uniform_pgrammar=True
    )
    print("done in", c.elapsed_time(), "s")
# We only get a task generator for the input generator
input_sampler = task_generator.input_generator

# ================================
# Load dataset & Task Generator
# ================================
simpler_pruner = UseAllVariablesPruner()

sampled_inputs = {}

syntaxic_restrictions: Dict[str, Set[str]] = defaultdict(set)
specific_restrictions = set()
pattern_constraints = []

stats = {
    s: {"total": 0, "syntaxic": 0}
    for s in ["identity", "symmetry", "constant", "equivalent", "invariant"]
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


def dump_primitives(program: Program) -> List[str]:
    pattern = []
    # Find all variables
    for p in program.depth_first_iter():
        if isinstance(p, Primitive):
            pattern.append(p.primitive)
    return pattern


def add_syntaxic(program: Program):
    pattern = dump_primitives(program)
    syntaxic_restrictions[pattern[0]].add(pattern[1])


def add_constraint_for(program: Program, category: str):
    prog_vars = vars(program)
    max_vars = len(prog_vars)
    stats[category]["total"] += 1
    # If only one variable used, this is easy
    if max_vars == 1:
        add_syntaxic(program)
        stats[category]["syntaxic"] += 1
        return

    if category == "symmetry":
        # global pattern_constraints
        variables = vars(program)
        varnos = [v.variable for v in variables]
        arguments = [v.type for v in sorted(variables, key=lambda k: k.variable)]
        rtype = program.type
        swap_indices = {
            varnos[i] != i and arg_type == rtype for i, arg_type in enumerate(arguments)
        }
        if len(swap_indices) == len(varnos):
            return

        primitive = dump_primitives(program)[0]
        pattern_constraints.append(
            f"{primitive} "
            + " ".join(
                SYMBOL_ANYTHING
                if i not in swap_indices
                else f"{SYMBOL_FORBIDDEN}{primitive}"
                for i in varnos
            )
        )

    else:
        global specific_restrictions
        specific_restrictions.add(program)


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
    iterable = (
        tqdm.tqdm(dsl.list_primitives, smoothing=1) if progress else dsl.list_primitives
    )

    all_solutions = defaultdict(dict)
    programs_done = set()
    # Pre Sample Inputs + Pre Execute base primitives
    for primitive in dsl.list_primitives:
        arguments = (
            [] if not isinstance(primitive.type, Arrow) else primitive.type.arguments()
        )
        if len(arguments) == 0:
            continue
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
            all_solutions[primitive.type.returns()][primitive] = solutions

    # Check for symmetry
    for primitive in iterable:
        arguments = (
            [] if not isinstance(primitive.type, Arrow) else primitive.type.arguments()
        )
        if len(arguments) == 0:
            continue
        if progress:
            iterable.set_postfix_str(primitive.primitive)

        base_program = Function(
            primitive,
            [Variable(i, arg_type) for i, arg_type in enumerate(arguments)],
        )
        inputs = sampled_inputs[primitive.type]
        solutions = all_solutions[primitive]

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
                programs_done.add(current_prog)
                is_symmetric = current_prog != base_program
                varno = args[0].variable
                for inp, sol in zip(inputs, solutions):
                    with chrono.clock("search.variants.enumeration.eval"):
                        out = evaluator.eval(current_prog, inp)
                    if is_symmetric and out != sol:
                        is_symmetric = False
                        break
                if is_symmetric:
                    add_constraint_for(current_prog, "symmetry")

    ftypes = tqdm.tqdm(sampled_inputs.keys()) if progress else sampled_inputs.keys()
    for ftype in ftypes:
        # Add forbidden patterns to speed up search
        dsl.forbidden_patterns = copy.deepcopy(syntaxic_restrictions)
        cfg = CFG.depth_constraint(dsl, ftype, max_depth + 1)
        pcfg = ProbDetGrammar.uniform(cfg)
        inputs = sampled_inputs[ftype]

        all_sol = all_solutions[ftype.returns()]

        cfg_size = cfg.size()
        done = 0
        if progress:
            ftypes.set_postfix_str(f"{done / cfg_size:.0%}")

        # ========================
        # Check all programs starting with max depth
        # ========================
        with chrono.clock("search.invariant.enumeration"):
            for program in enumerate_prob_grammar(pcfg):
                done += 1
                if program in programs_done:
                    continue
                is_constant = True
                my_outputs = []
                candidates = set(all_sol.keys())
                with chrono.clock("search.invariant.enumeration.pruning"):
                    if not simpler_pruner.accept((primitive.type, program)):
                        continue
                is_invariant = True
                is_identity = len(arguments) == 1
                for i, inp in enumerate(inputs):
                    with chrono.clock("search.invariant.enumeration.eval"):
                        out = evaluator.eval(program, inp)
                    # Update candidates
                    candidates = {c for c in candidates if all_sol[c][i] == out}
                    if is_identity and out != inp[0]:
                        is_identity = False
                    if is_constant and len(my_outputs) > 0 and my_outputs[-1] != out:
                        is_constant = False
                    my_outputs.append(out)
                    if (
                        not is_constant
                        and not is_identity
                        and len(candidates) == 0
                        and not in_depth_equivalence_check
                    ):
                        break
                if is_identity:
                    add_constraint_for(program, "identity")
                if is_constant:
                    add_constraint_for(program, "constant")
                if len(candidates) > 0:
                    if any(c == primitive for c in candidates):
                        add_constraint_for(program, "invariant")
                    else:
                        add_constraint_for(program, "equivalent")
                elif in_depth_equivalence_check and not is_identity and not is_constant:
                    all_sol[program] = my_outputs
                if done & 256 == 0:
                    ftypes.set_postfix_str(f"{done / cfg_size:.0%}")

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
        # Add forbidden patterns to speed up search
        dsl.forbidden_patterns = copy.deepcopy(syntaxic_restrictions)
        cfg = CFG.depth_constraint(dsl, primitive.type, max_depth + 1)
        pcfg = ProbDetGrammar.uniform(cfg)
        with chrono.clock("constants.enumeration"):
            for program in enumerate_prob_grammar(pcfg):
                with chrono.clock("constants.enumeration.eval"):
                    out = evaluator.eval(program, [])
                if out in all_evals:
                    constant_program_analysis(program)

specific_reused = 0
with chrono.clock("pattern"):
    with chrono.clock("pattern.encode"):
        all_found: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        for program in specific_restrictions:
            prims = tuple(dump_primitives(program))
            all_found[prims].add(tuple([v.variable for v in vars(program)]))
    with chrono.clock("pattern.find"):
        for derivations, all_variants in all_found.items():
            print("For", derivations, "=>", all_variants)
            nvars = len(list(all_variants)[0])
            good = True
            for comb in itertools.permutations(list(range(nvars))):
                if comb not in all_variants:
                    good = False
                    break
            if good:
                specific_reused += len(all_variants)
                pattern_constraints.append(0)


print(
    "Done",
    chrono.summary(
        time_formatter=lambda t: f"{int(t*1000)}ms" if not np.isnan(t) else "nan"
    ),
)
print(f"Cache hit rate: {evaluator.cache_hit_rate:.1%}")
print()
print("[=== Report ===]")
for stat_name in stats:
    total = stats[stat_name]["total"]
    if total == 0:
        print(f"Found no {stat_name} simplification")
        continue
    ratio = stats[stat_name]["syntaxic"] / total
    print("Found", total, stat_name, f"with {ratio:.1%} syntaxic")

print("[=== Results ===]")
print(f"Found {len(syntaxic_restrictions)} syntaxic restricions.")
copied_dsl = DSL(
    {p.primitive: p.type for p in dsl.list_primitives}, syntaxic_restrictions
)
max_depth = 4
all_type_requests = set(task_generator.type2pgrammar.keys())
dsl.forbidden_patterns = {}
cfgs = [CFG.depth_constraint(dsl, t, max_depth) for t in all_type_requests]
reduced_cfgs = [
    CFG.depth_constraint(copied_dsl, t, max_depth) for t in all_type_requests
]
ratios = [
    (original.size() - red.size()) / original.size()
    for original, red in zip(cfgs, reduced_cfgs)
]
print(
    f"At depth {max_depth}, it is an average reduction of {np.mean(ratios):.2%} [{np.min(ratios):.1%} -> {np.max(ratios):.1%}] of CFG size"
)
print("{", end="")
for k, v in sorted(syntaxic_restrictions.items()):
    print(f'"{k}": ', '{ "' + '", "'.join(sorted(v)) + '"},', end=" ")
print("}\n")
print(f"Found {len(pattern_constraints)} type constraints:")
print(pattern_constraints)
print(
    f"Found {len(specific_restrictions) - specific_reused} specific restricions that could not be added as constraint, impact not computed."
)
# print(specific_restrictions)
