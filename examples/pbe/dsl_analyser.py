from typing import Set

import numpy as np

from synth import Dataset, PBE
from synth.pbe import reproduce_dataset
from synth.pruning import UseAllVariablesPruner
from synth.syntax import ConcreteCFG, ConcretePCFG, enumerate_pcfg
from synth.syntax.dsl import DSL
from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow
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
semantic_restrictions = []

total_identity_candidates = 0
total_identity_candidates_kept = 0

total_redundant_candidates = 0
total_redundant_candidates_kept = 0

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
        pcfg = ConcretePCFG.uniform_from_cfg(cfg)

        with chrono.clock("search.input_sampling"):
            if primitive.type not in sampled_inputs:
                sampled_inputs[primitive.type] = [
                    [input_sampler.sample(type=arg) for arg in arguments]
                    for _ in range(input_checks)
                ]
            inputs = sampled_inputs[primitive.type]
        # ========================
        # Identity part
        # ========================
        if len(arguments) == 1:
            identity_candidates: Set[Program] = set()
            with chrono.clock("search.identity.enumeration"):
                for program in enumerate_pcfg(pcfg):
                    with chrono.clock("search.identity.enumeration.pruning"):
                        if not simpler_pruner.accept((primitive.type, program)):
                            continue
                    is_identity = True
                    for inp in inputs:
                        with chrono.clock("search.identity.enumeration.eval"):
                            out = evaluator.eval(program, inp)
                        if out != inp[0]:
                            is_identity = False
                            break
                    if is_identity:
                        identity_candidates.add(program)

            total_identity_candidates += len(identity_candidates)
            with chrono.clock("search.identity.validation"):
                for candidate in identity_candidates:
                    variables = []
                    pattern = []
                    # Find all variables
                    for p in candidate.depth_first_iter():
                        if isinstance(p, Variable):
                            variables.append(p)
                        elif isinstance(p, Primitive):
                            pattern.append(p.primitive)
                    # If only one variable was used, we can add a syntaxic restriction
                    if len(variables) == 1:
                        total_identity_candidates_kept += 1
                        syntaxic_restrictions.append(pattern)
                    else:
                        semantic_restrictions.append(candidate)

        # ========================
        # Redundant part
        # ========================
        with chrono.clock("search.redundant.compute_solutions"):
            prog = Function(
                primitive,
                [Variable(i, arg_type) for i, arg_type in enumerate(arguments)],
            )
            solutions = [evaluator.eval(prog, inp) for inp in inputs]

        redundant_candidates: Set[Program] = set()
        with chrono.clock("search.redundant.enumeration"):
            for program in enumerate_pcfg(pcfg):
                if prog == program:
                    continue
                with chrono.clock("search.redundant.enumeration.pruning"):
                    if not simpler_pruner.accept((primitive.type, program)):
                        continue
                is_redundant = True
                for inp, sol in zip(inputs, solutions):
                    with chrono.clock("search.redundant.enumeration.eval"):
                        out = evaluator.eval(program, inp)
                    if out != sol:
                        is_redundant = False
                        break
                    is_redundant &= out == sol

                if is_redundant:
                    redundant_candidates.add(program)
        total_redundant_candidates += len(redundant_candidates)
        with chrono.clock("search.redundant.validation"):
            for candidate in redundant_candidates:
                variables = []
                pattern = []
                # Find all variables
                for p in candidate.depth_first_iter():
                    if isinstance(p, Variable):
                        variables.append(p)
                    elif isinstance(p, Primitive):
                        pattern.append(p.primitive)
                # If only one variable was used, we can add a syntaxic restriction
                if len(variables) == 1:
                    total_redundant_candidates_kept += 1
                    syntaxic_restrictions.append(pattern)
                else:
                    semantic_restrictions.append(candidate)
print(
    "Done",
    chrono.summary(
        time_formatter=lambda t: f"{int(t*1000)}ms" if not np.isnan(t) else "nan"
    ),
)
print()
print(
    "Total identity candidates:",
    total_identity_candidates,
    "kept:",
    total_identity_candidates_kept,
    f"({100 * total_identity_candidates_kept / total_identity_candidates:.1f}%)",
)
print(
    "Total redundant candidates:",
    total_redundant_candidates,
    "kept:",
    total_redundant_candidates_kept,
    f"({100 * total_redundant_candidates_kept / total_redundant_candidates:.1f}%)",
)
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
