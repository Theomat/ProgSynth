from typing import Optional

from colorama import Fore as F

from synth import Dataset, PBE
from synth.syntax import CFG, Program
from synth.task import Task

from dsl_loader import add_dsl_choice_arg, load_DSL
from dataset_loader import add_dataset_choice_arg, load_dataset

import argparse

parser = argparse.ArgumentParser(description="Explore a dataset")
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
# ================================
# Load constants specific to DSL
# ================================
def pretty_print_solution(program: Program):
    return "\n".join(program.pretty_print())


def pretty_print_inputs(str: str):
    return str


dsl_module = load_DSL(dsl_name)
dsl, lexicon = dsl_module.dsl, dsl_module.lexicon
if hasattr(dsl_module, "pretty_print_solution"):
    pretty_print_solution = dsl_module.pretty_print_solution
if hasattr(dsl_module, "pretty_print_inputs"):
    pretty_print_inputs = dsl_module.pretty_print_inputs
# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
full_dataset: Dataset = load_dataset(dsl_name, dataset_file)


def print_value(name: str, value: str) -> None:
    print(f"{F.GREEN}{name}{F.RESET}: {F.LIGHTYELLOW_EX}{value}{F.RESET}")


def summary(*args: str) -> None:
    all_type_requests = full_dataset.type_requests()
    print(
        f"{F.LIGHTYELLOW_EX}{len(full_dataset)} {F.GREEN}tasks{F.RESET}, {F.LIGHTYELLOW_EX}{len([task for task in full_dataset if task.solution]) / len(full_dataset):.1%}{F.RESET} of which have {F.GREEN}solutions{F.RESET}."
    )
    print(
        f"{F.LIGHTYELLOW_EX}{len(all_type_requests)} {F.GREEN}type requests{F.RESET} supported."
    )
    print_value("Lexicon", f"[{min(lexicon)};{max(lexicon)}]")


def types(*args: str) -> None:
    all_type_requests = full_dataset.type_requests()
    total = len(full_dataset)
    max_len = max([len(str(t)) for t in all_type_requests])
    for type_req in all_type_requests:
        n = len([task for task in full_dataset if task.type_request == type_req])
        percent = f"{n / total:.1%}"
        print_value(f"{type_req!s:<{max_len}}", f"({percent:>5}) {n}")


def cfg(*args: str) -> None:
    max_depth = 4
    if args:
        try:
            max_depth = int(args[0])
        except:
            pass
    all_type_requests = full_dataset.type_requests()
    print_value("Max Depth", max_depth)
    max_len = max([len(str(t)) for t in all_type_requests])
    programs_no = {
        t: f"{CFG.depth_constraint(dsl, t, max_depth).programs():,}"
        for t in all_type_requests
    }
    max_len_programs_no = max(len(s) for s in programs_no.values())
    print_value(
        "{0:<{max_len}}".format("<type>", max_len=max_len),
        "{0:>{max_len_programs_no}}   {1}".format(
            "<programs>",
            "<rules>",
            max_len_programs_no=max_len_programs_no,
        ),
    )
    for type_req in all_type_requests:
        cfg = CFG.depth_constraint(dsl, type_req, max_depth)
        print_value(
            f"{type_req!s:<{max_len}}",
            f"{programs_no[type_req]:>{max_len_programs_no}}   {len(cfg.rules)}",
        )


def task(*args: str) -> None:
    try:
        task_no = int(args[0])
    except:
        print(
            f"{F.LIGHTRED_EX}You must specify a task number in the range[0;{len(full_dataset) - 1}]!{F.RESET}"
        )
        return
    if task_no < 0 or task_no >= len(full_dataset):
        print(
            f"{F.LIGHTRED_EX}{task_no} is an invalid task number, it must be in the range [0;{len(full_dataset) - 1}]!{F.RESET}"
        )
        return
    task: Task[PBE] = full_dataset[task_no]
    print_value(f"Name", task.metadata.get("name", "None"))
    print_value("Type", task.type_request)
    if task.solution is not None:
        print_value("Solution", "\n" + pretty_print_solution(task.solution))
    else:
        print(f"{F.LIGHTRED_EX}No Solution{F.RESET}")
    print_value("Examples", "")
    for example in task.specification.examples:
        print_value(
            "\tInput",
            ", ".join(
                [
                    f"var{i}={pretty_print_inputs(x)}"
                    for i, x in enumerate(example.inputs)
                ]
            ),
        )
        print_value("\tOutput", example.output)
        print()
    print_value("Metadata", task.metadata)


def filter_tasks(*args: str) -> None:
    if not args:
        print(
            F.LIGHTRED_EX
            + "Invalid syntax: you must give a valid boolean python expression that only depend on a task parameter."
            + F.RESET
        )
        return
    code = "[i for i, task in enumerate(full_dataset) if " + " ".join(args) + "]"
    queried_tasks = eval(code)
    if len(queried_tasks) == 0:
        print(f"{F.LIGHTYELLOW_EX}No {F.GREEN}task{F.RESET} matched your query!")
    elif len(queried_tasks) == 1:
        print(
            f"{F.GREEN}Task {F.LIGHTYELLOW_EX}n°{queried_tasks[0]}{F.RESET} matched your query!"
        )
    else:
        print(
            f"{F.LIGHTYELLOW_EX}{len(queried_tasks)} {F.GREEN}tasks{F.RESET} matched your query:"
        )
        print(queried_tasks)


COMMANDS = {
    "summary": summary,
    "types": types,
    "task": task,
    "cfg": cfg,
    "lexicon": lambda *args: print(lexicon),
    "filter": filter_tasks,
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
        print(
            "Available commands are:",
            ", ".join([F.GREEN + x + F.RESET for x in COMMANDS.keys()]) + ".",
        )


print(f'Type "{F.GREEN}help{F.RESET}" for help.')
print(
    f'Unambigous commands prefixes work (e.g: "{F.LIGHTGREEN_EX}h{F.RESET}" for "{F.GREEN}help{F.RESET}").'
)
while True:
    try:
        cmd = input(">").lower().strip()
    except EOFError:
        break
    try_execute_command(cmd)
