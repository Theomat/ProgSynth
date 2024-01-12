from collections import defaultdict
import json
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TypeVar
import copy
import argparse

import tqdm
import numpy as np

from synth.generation.sampler import (
    LexiconSampler,
    RequestSampler,
    Sampler,
    UnionSampler,
)
from synth.filter import UseAllVariablesPruner
from synth.syntax import (
    CFG,
    ProbDetGrammar,
    bps_enumerate_prob_grammar as enumerate_prob_grammar,
    Function,
    Primitive,
    Program,
    Variable,
    Arrow,
    Type,
    auto_type,
)

from karel import dsl, evaluator, KarelWorld
from karel_task_generator import random_world


parser = argparse.ArgumentParser(description="Generate equations for Karel")
parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
parser.add_argument(
    "--n", type=int, default=50, help="number of examples to be sampled (default: 50)"
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=5,
    help="max depth of programs to check for (default: 5)",
)


parameters = parser.parse_args()
dsl_name: str = "karel"
input_checks: int = parameters.n
max_depth: int = parameters.max_depth
seed: int = parameters.seed
# ================================
# Initialisation
# ================================

our_eval = lambda *args: evaluator.eval(*args)


class GridSampler(RequestSampler[KarelWorld]):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def sample_for(self, type: Type, **kwargs: Any) -> KarelWorld:
        return random_world(10, 10, self.rng)


basic_samplers: Dict[Type, Sampler] = {auto_type("world"): GridSampler(seed)}
for this_type in [auto_type("stmt"), auto_type("cond"), auto_type("int")]:
    elements = []
    for el in dsl.list_primitives:
        if el.type == this_type:
            elements.append(evaluator.semantics[el.primitive])
    basic_samplers[this_type] = LexiconSampler(elements, seed=seed)

# t = auto_type("stmt")
# stmt_lex = []
# comp: List[Primitive] = []
# for el in dsl.list_primitives:
#     if el.type == t:
#         stmt_lex.append(evaluator.semantics[el.primitive])
#     elif el.type.returns() == t:
#         comp.append(el)

# for _ in range(1):
#     print("level:", _, "statements:", len(stmt_lex))
#     current_gen = []
#     for f in comp:
#         candidates = []
#         for arg_t in f.type.arguments():
#             if arg_t == t:
#                 candidates.append(stmt_lex)
#             else:
#                 candidates.append(basic_samplers[arg_t].lexicon)
#         for poss in product(*candidates):
#             fun = evaluator.semantics[f.primitive]
#             args = list(poss)
#             value = fun
#             while args:
#                 value = value(args.pop(0))
#             current_gen.append(value)

#     stmt_lex += current_gen
# basic_samplers[t] = LexiconSampler(stmt_lex, seed=seed)
input_sampler = UnionSampler(basic_samplers)

# ================================
# Load dataset & Task Generator
# ================================
syntaxic_restrictions: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
specific_restrictions = set()
pattern_constraints = []

symmetrics: List[Tuple[str, Set[int]]] = []


stats = {s: {"total": 0, "syntaxic": 0} for s in ["identity", "constant", "equivalent"]}


# =========================================================================================
# Equivalence classes
# =========================================================================================

equivalence_classes = defaultdict(dict)
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


# =========================================================================================
# =========================================================================================


def forbid_first_derivation(program: Program) -> bool:
    pattern = dump_primitives(program)
    if len(pattern) != 2:
        return False
    syntaxic_restrictions[(pattern[0], 0)].add(pattern[1])
    return True


def add_symmetric(program: Program) -> None:
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
    symmetrics.append((primitive, swap_indices))


def add_constraint_for(program: Program, category: str):
    prog_vars = vars(program)
    max_vars = len(prog_vars)
    stats[category]["total"] += 1
    # If only one variable used, this is easy
    if max_vars == 1 and forbid_first_derivation(program):
        stats[category]["syntaxic"] += 1
        return
    global specific_restrictions
    specific_restrictions.add(program)


def constant_program_analysis(program: Program):
    category = "constant"
    prog_consts = constants(program)
    max_consts = len(prog_consts)
    stats[category]["total"] += 1

    # If only one constant used, this is easy
    if max_consts == 1 and forbid_first_derivation(program):
        stats[category]["syntaxic"] += 1
        return
    else:
        # TODO: this may be done better
        global specific_restrictions
        specific_restrictions.add(program)


# =========================================================================================
# =========================================================================================

sampled_inputs = {}
all_solutions = defaultdict(dict)
programs_done = set()
forbidden_types = set()


def init_base_primitives() -> None:
    primitives: List[Primitive] = dsl.list_primitives
    # Check forbidden types
    all_types = set()
    for primitive in primitives:
        all_types |= primitive.type.decompose_type()[0]
    for arg_type in all_types:
        try:
            input_sampler.sample(type=arg_type)
        except Exception as e:
            forbidden_types.add(arg_type)
            # raise e
    print("Some types could not be sampled:", forbidden_types)
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
        solutions = [our_eval(base_program, inp) for inp in inputs]
        all_solutions[base_program.type.returns()][base_program] = solutions
        new_equivalence_class(base_program)


def check_symmetries() -> None:
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
        solutions = all_sol[base_program]

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
            programs_done.add(current_prog)
            is_symmetric = current_prog != base_program
            outputs = []
            for inp, sol in zip(inputs, solutions):
                out = our_eval(current_prog, inp)
                outputs.append(out)
                if is_symmetric and out != sol:
                    is_symmetric = False
            if is_symmetric:
                add_symmetric(current_prog)
                merge_equivalence_classes(current_prog, base_program)
            else:
                new_equivalence_class(base_program)
                all_sol[current_prog] = outputs


def check_equivalent() -> None:
    simpler_pruner = UseAllVariablesPruner()
    ftypes = tqdm.tqdm(sampled_inputs.keys())
    for ftype in ftypes:
        # Add forbidden patterns to speed up search
        dsl.forbidden_patterns = copy.deepcopy(syntaxic_restrictions)
        cfg = CFG.depth_constraint(dsl, ftype, max_depth + 1)
        pcfg = ProbDetGrammar.uniform(cfg)

        inputs = sampled_inputs[ftype]
        all_sol = all_solutions[ftype.returns()]

        cfg_size = cfg.programs()
        print("programs:", cfg_size)
        ftypes.set_postfix_str(f"{0 / cfg_size:.0%}")

        # ========================
        # Check all programs starting with max depth
        # ========================
        for done, program in enumerate(enumerate_prob_grammar(pcfg)):
            if program in programs_done or not simpler_pruner.accept((ftype, program)):
                continue
            ftypes.set_postfix_str(f"{done}/{cfg_size} | {done / cfg_size:.0%}")
            is_constant = True
            my_outputs = []
            candidates = set(all_sol.keys())
            is_identity = len(program.used_variables()) == 1
            for i, inp in enumerate(inputs):
                out = our_eval(program, inp)
                # Update candidates
                candidates = {c for c in candidates if all_sol[c][i] == out}
                if is_identity and out != inp[0]:
                    is_identity = False
                if is_constant and len(my_outputs) > 0 and my_outputs[-1] != out:
                    is_constant = False
                my_outputs.append(out)
            if is_identity:
                add_constraint_for(program, "identity")
            elif is_constant:
                add_constraint_for(program, "constant")
            elif len(candidates) > 0:
                merge_equivalence_classes(program, list(candidates)[0])
                add_constraint_for(program, "equivalent")
            else:
                new_equivalence_class(program)
                all_sol[program] = my_outputs
            # if done & 256 == 0:
            #     ftypes.set_postfix_str(f"{done / cfg_size:.0%}")


def check_constants() -> None:
    types = set(
        primitive.type
        for primitive in dsl.list_primitives
        if not isinstance(primitive.type, Arrow)
    )
    for ty in types:
        all_evals = {our_eval(p, []) for p in dsl.list_primitives if p.type == ty}
        # Add forbidden patterns to speed up search
        dsl.forbidden_patterns = copy.deepcopy(syntaxic_restrictions)
        cfg = CFG.depth_constraint(dsl, ty, max_depth + 1)
        pcfg = ProbDetGrammar.uniform(cfg)
        for program in enumerate_prob_grammar(pcfg):
            out = our_eval(program, [])
            if out in all_evals:
                constant_program_analysis(program)


def exploit_symmetries() -> None:
    sym_types = defaultdict(set)
    name2P = {}
    for P in dsl.list_primitives:
        for name, _ in symmetrics:
            if P.primitive == name:
                sym_types[P.type.returns()].add(name)
                name2P[name] = P
                break
    for name, forbid_indices in symmetrics:
        P: Primitive = name2P[name]
        for i, arg in enumerate(P.type.arguments()):
            if i in forbid_indices:
                syntaxic_restrictions[(P.primitive, i)] |= sym_types[arg]


print("Primitives:")
init_base_primitives()
print("Symmetries:")
check_symmetries()
print("Equivalents:")
check_equivalent()
print("Constants:")
check_constants()
print("Exploiting symmetries:")
exploit_symmetries()

print(f"Cache hit rate: {evaluator.cache_hit_rate:.1%}")
print()
print("[=== Report ===]")
for stat_name in stats:
    total = stats[stat_name]["total"]
    if total == 0:
        print(f"Found no {stat_name} program.")
        continue
    ratio = stats[stat_name]["syntaxic"] / total
    print(
        "Found",
        total,
        stat_name,
        f"{ratio:.1%} of which were translated into constraints.",
    )
if len(symmetrics) > 0:
    print(
        f"Found {len(symmetrics)} symmetries of which 100% were translated into constraints."
    )
else:
    print("Found no symmetries.")
print("[=== Results ===]")
print(
    f"Produced {sum(len(x) for x in syntaxic_restrictions.values())} forbidden patterns."
)
print(f"Produced {len(pattern_constraints)} type constraints.")

# Saving
with open(f"constraints_{dsl_name}.py", "w") as fd:
    fd.write("forbidden_patterns = {")
    for k, v in sorted(syntaxic_restrictions.items()):
        fd.write(f'{k}:  {{ "' + '", "'.join(sorted(v)) + '"}, ')
    fd.write("}\n")
    fd.write("\n")
    fd.write("pattern_constraints = ")
    fd.write(str(pattern_constraints))

classes = [get_equivalence_class(i) for i in range(n_equiv_classes)]
classes = [l for l in classes if len(l) > 1]
with open(f"equivalent_classes_{dsl_name}.json", "w") as fd:
    my_list = [list(map(str, l)) for l in classes]
    json.dump(my_list, fd)

print(
    f"Found {len(classes)} equivalence classes of which 0% were translated into constraints."
)
print(
    f"Produced:\n\t- constraints_{dsl_name}.py: which contains the encoded constraints\n\t- equivalent_classes_{dsl_name}.json: which contains the equivalence classes"
)
