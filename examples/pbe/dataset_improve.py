import argparse
import csv

from dsl_loader import add_dsl_choice_arg, load_DSL
from dataset_loader import add_dataset_choice_arg, load_dataset

from synth import Dataset, PBE
from synth.utils import chrono


parser = argparse.ArgumentParser(
    description="Generate a new dataset by replacing solutions found if they are shorter than the original's"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "-s",
    "--solution",
    type=str,
    help="solution file",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
solution_file: str = parameters.solution
# ================================
# Load constants specific to DSL
# ================================
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon
# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
full_dataset: Dataset[PBE] = load_dataset(dsl_name, dataset_file)

print("Loading solutions...", end="", flush=True)
with chrono.clock("solutions.load") as c:
    with open(solution_file, "r") as fd:
        reader = csv.reader(fd)
        trace = [tuple(row) for row in reader]
        trace.pop(0)
        solutions = [row[-1] if row[0] == "True" else None for row in trace]
    print("done in", c.elapsed_time(), "s")

replaced = 0
saved = 0
print("Merging solutions and dataset...", end="", flush=True)
with chrono.clock("merge") as c:
    for task, new_sol in zip(full_dataset.tasks, solutions):
        if new_sol is None:
            continue
        if task.solution is None:
            task.solution = dsl.parse_program(new_sol, task.type_request)
            continue
        size = new_sol.count(" ") + 1
        if size < task.solution.size():
            saved += task.solution.size() - size
            task.solution = dsl.parse_program(new_sol, task.type_request)
            replaced += 1

    print("done in", c.elapsed_time(), "s")
print(f"Replaced {replaced} original solutions saving {saved} size!")
print("Saving merged dataset...", end="", flush=True)
with chrono.clock("dataset.save") as c:
    full_dataset.save(dataset_file.replace(".pickle", "_merged.pickle"))
    print("done in", c.elapsed_time(), "s")
