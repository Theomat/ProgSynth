import os
import sys
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import csv

import tqdm

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL


from synth import Dataset, PBE
from synth.semantic.evaluator import DSLEvaluator
from synth.specification import PBEWithConstants
from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    DSL,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    hs_enumerate_bucket_prob_grammar,
    hs_enumerate_bucket_prob_u_grammar,
    ProgramEnumerator,
)
from synth.utils import load_object
from synth.pbe.solvers import (
    NaivePBESolver,
    PBESolver,
    CutoffPBESolver,
    ObsEqPBESolver,
    RestartPBESolver,
)


import argparse


SOLVERS = {
    solver.name(): solver
    for solver in [NaivePBESolver, CutoffPBESolver, ObsEqPBESolver]
}
base_solvers = {x: y for x, y in SOLVERS.items()}
for meta_solver in [RestartPBESolver]:
    for name, solver in base_solvers.items():
        SOLVERS[f"{meta_solver.name()}.{name}"] = lambda *args, **kwargs: meta_solver(
            *args, solver_builder=solver, **kwargs
        )

SEARCH_ALGOS = {
    "beap_search": (bps_enumerate_prob_grammar, None),
    "heap_search": (hs_enumerate_prob_grammar, hs_enumerate_prob_u_grammar),
    "bucket_search": (
        lambda x: hs_enumerate_bucket_prob_grammar(x, 3),
        lambda x: hs_enumerate_bucket_prob_u_grammar(x, 3),
    ),
    "bee_search": (bs_enumerate_prob_grammar, None),
}

parser = argparse.ArgumentParser(
    description="Solve program synthesis tasks", fromfile_prefix_chars="@"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument("--pcfg", type=str, help="files containing the predicted PCFGs")
parser.add_argument(
    "-s",
    "--search",
    choices=SEARCH_ALGOS.keys(),
    default=list(SEARCH_ALGOS.keys())[0],
    help=f"enumeration algorithm (default: {list(SEARCH_ALGOS.keys())[0]})",
)
parser.add_argument(
    "--solver",
    choices=list(SOLVERS.keys()),
    default="naive",
    help=f"used solver (default: naive)",
)
parser.add_argument(
    "-o", "--output", type=str, default="./", help="output folder (default: './')"
)
parser.add_argument(
    "--constrained",
    action="store_true",
    default=False,
    help="use unambigous grammar to include constraints in the grammar if available",
)
parser.add_argument(
    "--support",
    type=str,
    default=None,
    help="train dataset to get the set of supported type requests",
)
parser.add_argument(
    "-t", "--timeout", type=float, default=300, help="task timeout in s (default: 300)"
)


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
pcfg_file: str = parameters.pcfg
search_algo: str = parameters.search
method: Callable[[Any], PBESolver] = SOLVERS[parameters.solver]
output_folder: str = parameters.output
task_timeout: float = parameters.timeout
constrained: bool = parameters.constrained
support: Optional[str] = (
    None if not parameters.support else parameters.support.format(dsl_name=dsl_name)
)

if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)
elif support is not None and (
    not os.path.exists(support) or not os.path.isfile(support)
):
    print("Support dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)

det_search, u_search = SEARCH_ALGOS[search_algo]
custom_enumerate = u_search if constrained else det_search
if custom_enumerate is None:
    txt = "det-CFG" if not constrained else "UCFG"
    print(
        f"search algorithm {search_algo} does not support enumeration for {txt}!",
        file=sys.stderr,
    )
    sys.exit(1)

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

supported_type_requests = Dataset.load(support).type_requests() if support else None

# ================================
# Load constants specific to dataset
# ================================


def load_dsl_and_dataset() -> Tuple[Dataset[PBE], DSL, DSLEvaluator]:
    dsl_module = load_DSL(dsl_name)
    dsl, evaluator = dsl_module.dsl, dsl_module.evaluator
    # ================================
    # Load dataset
    # ================================
    full_dataset = load_dataset(dsl_name, dataset_file)

    return full_dataset, dsl, evaluator


# Produce PCFGS ==========================================================


def save(trace: Iterable) -> None:
    with open(file, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(trace)


# Enumeration methods =====================================================
def enumerative_search(
    dataset: Dataset[PBE],
    evaluator: DSLEvaluator,
    pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]],
    trace: List[Tuple[bool, float]],
    solver: PBESolver,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
) -> None:

    start = max(0, len(trace) - 1)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks", smoothing=0)
    i = 0
    solved = 0
    total = 0
    tasks = [
        t
        for t in dataset.tasks
        if supported_type_requests is None or t.type_request in supported_type_requests
    ]
    stats_name = solver.available_stats()
    if start == 0:
        trace.append(["solved", "solution"] + stats_name)
    for task, pcfg in zip(tasks[start:], pcfgs[start:]):
        if task.metadata.get("name", None) is not None:
            pbar.set_description_str(task.metadata["name"])
        total += 1
        task_solved = False
        solution = None
        if isinstance(task.specification, PBEWithConstants):
            pcfg = pcfg.instantiate_constants(task.specification.constants)
        try:
            sol_generator = solver.solve(
                task, custom_enumerate(pcfg), timeout=task_timeout
            )
            solution = next(sol_generator)
            task_solved = True
            solved += 1
        except KeyboardInterrupt:
            break
        except StopIteration:
            pass
        out = [task_solved, solution] + [solver.get_stats(name) for name in stats_name]
        solver.reset_stats()
        trace.append(out)
        pbar.update(1)
        evaluator.clear_cache()
        # print("Cache hit:", evaluator.cache_hit_rate)
        # print("Programs tried:", trace[len(trace) - 1][2])
        if i % 10 == 0:
            pbar.set_postfix_str("Saving...")
            save(trace)
        pbar.set_postfix_str(f"Solved {solved}/{total}")

    pbar.close()


# Main ====================================================================

if __name__ == "__main__":
    (
        full_dataset,
        dsl,
        evaluator,
    ) = load_dsl_and_dataset()

    solver: PBESolver = method(evaluator=evaluator)

    pcfgs = load_object(pcfg_file)
    model_name = os.path.split(pcfg_file)[1][
        len(f"pcfgs_{dataset_name}_") : -len(".pickle")
    ]
    file = os.path.join(
        output_folder,
        f"{dataset_name}_{search_algo}_{model_name}_{solver.full_name()}.csv",
    )
    trace = []
    if os.path.exists(file):
        with open(file, "r") as fd:
            reader = csv.reader(fd)
            trace = [tuple(row) for row in reader]
            print(
                "\tLoaded",
                len(trace) - 1,
                "/",
                len(full_dataset),
                "(",
                int((len(trace) - 1) * 100 / len(full_dataset)),
                "%)",
            )
    enumerative_search(full_dataset, evaluator, pcfgs, trace, solver, custom_enumerate)
    save(trace)
    print("csv file was saved as:", file)
