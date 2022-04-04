import argparse
import csv
import sys
import typing

from synth import Dataset, PBE
from synth.syntax.program import Function, Program, Variable
from synth.syntax.type_system import Arrow, Type
from synth.utils import chrono

DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"
CALCULATOR = "calculator"


parser = argparse.ArgumentParser(
    description="Generate a new dataset by replacing solutions found if they are shorter than the original's"
)
parser.add_argument(
    "--dsl",
    type=str,
    default=DEEPCODER,
    help="dsl (default: deepcoder)",
    choices=[DEEPCODER, DREAMCODER, CALCULATOR],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="{dsl_name}.pickle",
    help="dataset file (default: {dsl_name}.pickle)",
)
parser.add_argument(
    "-s",
    "--solution",
    type=str,
    help="solution file",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset.format(dsl_name=dsl_name)
solution_file: str = parameters.solution
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
if dsl_name == DEEPCODER:
    from deepcoder.deepcoder import dsl, evaluator, lexicon

elif dsl_name == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, evaluator, lexicon

    max_list_length = 10
elif dsl_name == CALCULATOR:
    from calculator.calculator import dsl, evaluator, lexicon
else:
    print("Unknown dsl:", dsl_name, file=sys.stderr)
    sys.exit(1)
# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
print(f"Loading {dataset_file}...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load(dataset_file)
    print("done in", c.elapsed_time(), "s")


print("Loading solutions...", end="", flush=True)
with chrono.clock("solutions.load") as c:
    with open(solution_file, "r") as fd:
        reader = csv.reader(fd)
        trace = [tuple(row) for row in reader]
        trace.pop(0)
        solutions = [row[-1] if row[0] == "True" else None for row in trace]
    print("done in", c.elapsed_time(), "s")


def parse(program: str, type_request: Type) -> Program:
    if " " in program:
        parts = list(map(lambda p: parse(p, type_request), program.split(" ")))

        def parse_stack(l: typing.List[Program]) -> Program:
            if len(l) == 1:
                return l[0]
            current = l.pop(0)
            if isinstance(current.type, Arrow):
                args = [parse_stack(l) for arg in current.type.arguments()]
                return Function(current, args)
            return current

        sol = parse_stack(parts)
        assert str(sol) == program
        return sol
    else:
        program = program.strip("()")
        for P in dsl.list_primitives:
            if P.primitive == program:
                return P
        if program.startswith("var"):
            varno = int(program[3:])
            vart = type_request
            if isinstance(type_request, Arrow):
                vart = type_request.arguments()[varno]
            return Variable(varno, vart)
        assert False, f"can't parse: {program}"


replaced = 0
saved = 0
print("Merging solutions and dataset...", end="", flush=True)
with chrono.clock("merge") as c:
    for task, new_sol in zip(full_dataset.tasks, solutions):
        if new_sol is None:
            continue
        if task.solution is None:
            task.solution = parse(new_sol, task.type_request)
            continue
        size = new_sol.count(" ") + 1
        if size < task.solution.length():
            saved += task.solution.length() - size
            task.solution = parse(new_sol, task.type_request)
            replaced += 1

    print("done in", c.elapsed_time(), "s")
print(f"Replaced {replaced} original solutions saving {saved} size!")
print("Saving merged dataset...", end="", flush=True)
with chrono.clock("dataset.save") as c:
    full_dataset.save(dataset_file.replace(".pickle", "_merged.pickle"))
    print("done in", c.elapsed_time(), "s")
