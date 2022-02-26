import sys
from typing import Optional

from synth import Dataset, PBE
from synth.syntax import ConcreteCFG
from synth.task import Task
from synth.utils import chrono

DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"


import argparse

parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
)
parser.add_argument(
    "--dsl",
    type=str,
    default=DEEPCODER,
    help="dsl (default: deepcoder)",
    choices=[DEEPCODER, DREAMCODER],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="{dsl_name}.pickle",
    help="dataset file (default: {dsl_name}.pickle)",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset.format(dsl_name=dsl_name)
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
if dsl_name == DEEPCODER:
    from deepcoder.deepcoder import dsl, lexicon

elif dsl_name == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, lexicon

    max_list_length = 10
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


def summary(*args: str) -> None:
    all_type_requests = full_dataset.type_requests()
    print(
        f"{len(full_dataset)} tasks, {len([task for task in full_dataset if task.solution]) / len(full_dataset) * 100:.1f}% of which have solutions."
    )
    print(f"{len(all_type_requests)} type requests supported.")
    print(f"Lexicon: [{min(lexicon)};{max(lexicon)}]")


def types(*args: str) -> None:
    all_type_requests = full_dataset.type_requests()
    total = len(full_dataset)
    for type_req in all_type_requests:
        n = len([task for task in full_dataset if task.type_request == type_req])
        print(type_req, ":", n, f"({n * 100 / total:.1f}%)")


def pcfg(*args: str) -> None:
    max_depth = 4
    if args:
        try:
            max_depth = int(args[0])
        except:
            pass
    all_type_requests = full_dataset.type_requests()
    for type_req in all_type_requests:
        cfg = ConcreteCFG.from_dsl(dsl, type_req, max_depth)
        print(
            type_req,
            ": at depth",
            max_depth,
            "the CFG contains",
            cfg.size(),
            "programs with",
            len(cfg.rules),
            "contexts",
        )


def task(*args: str) -> None:
    try:
        task_no = int(args[0])
    except:
        print("You must specify a task number!")
        return
    if task_no < 0 or task_no >= len(full_dataset):
        print(
            f"{task_no} is an invalid task number, it must be in the range [0;{len(full_dataset)}]!"
        )
        return
    task: Task[PBE] = full_dataset[task_no]
    print("Name:", task.metadata.get("name", "None"))
    print("Type:", task.type_request)
    print("Solution:", task.solution)
    print("Examples:")
    for example in task.specification.examples:
        print(
            "\tInput:", ", ".join([f"var{i}={x}" for i, x in enumerate(example.inputs)])
        )
        print("\tOutput:", example.output)
        print()
    print("Metadata:", task.metadata)


COMMANDS = {
    "summary": summary,
    "types": types,
    "task": task,
    "pcfg": pcfg,
    "lexicon": lambda *args: print(lexicon),
}


def __expand_name__(prefix: str) -> Optional[str]:
    candidates = list(COMMANDS.keys())
    for i, l in enumerate(prefix):
        candidates = [cand for cand in candidates if len(cand) > i and cand[i] == l]
        if len(candidates) == 1:
            return candidates[0]
    for cand in candidates:
        if len(cand) == len(prefix):
            return cand
    return None


def try_execute_command(cmd: str) -> None:
    words = cmd.split(" ")
    real_cmd = __expand_name__(words.pop(0))
    if real_cmd:
        COMMANDS[real_cmd](*words)
    else:
        print("Available commands are:", ", ".join(COMMANDS.keys()))


while True:
    try:
        cmd = input("What would you like to know?\n>").lower().strip()
    except EOFError:
        break
    try_execute_command(cmd)
