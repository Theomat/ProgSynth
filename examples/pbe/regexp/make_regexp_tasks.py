import json
from typing import Any, Callable, Dict, Tuple, List as TList

import tqdm

from synth import Task, Dataset, PBE, Example
from synth.syntax import (
    STRING,
    FunctionType,
    List,
    Type,
    Function,
    Primitive,
    Program,
    Variable,
)

from type_regex import REGEXP
from regexp_dsl import dsl, evaluator
from synth.syntax.program import Constant, Lambda


name2type = {p.primitive: p.type for p in dsl.list_primitives}


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = sum(1 for t in tasks if t.solution)
    print(
        f"Converted {len(tasks)} tasks {int(100 * sols / len(tasks))}% containing solutions"
    )
    # Integrity check
    for task in tqdm.tqdm(tasks, desc="integrity check"):
        for ex in task.specification.examples:
            assert evaluator.eval(task.solution, ex.inputs) == ex.output


def convert_regexp(
    file: str = "dataset/new_data.json",
    output_file: str = "regexp.pickle",
) -> None:
    def load() -> Dataset[PBE]:
        tasks: TList[Task[PBE]] = []
        with open(file, "r") as fd:
            raw_tasks: TList[Dict[str, Any]] = json.load(fd)
            for raw_task in tqdm.tqdm(raw_tasks, desc="converting"):
                name: str = raw_task["program"]
                raw_examples: TList[Dict[str, Any]] = raw_task["examples"]
                inputs = [raw_example["inputs"] for raw_example in raw_examples]
                outputs: TList = [raw_example["output"] for raw_example in raw_examples]
                prog, type_request = __regexp_str2prog(name)
                examples = [
                    Example([list(x) for x in inp], out)
                    for inp, out in zip(inputs, outputs)
                    if out is not None
                ]
                if len(examples) < len(inputs):
                    continue
                tasks.append(
                    Task[PBE](type_request, PBE(examples), prog, {"name": name})
                )
        return Dataset(tasks, metadata={"dataset": "regexp", "source:": file})

    __convert__(load, output_file)


def __regexp_str2prog(s: str) -> Tuple[Program, Type]:
    parts = s.split("|")
    stack: TList[Program] = []
    var: int = 0
    type_stack: TList[Type] = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        # composition of methods
        if name == "eval":
            primitive = Primitive(name, name2type[name])
            targets = [int(x) for x in subparts]
            arguments = [stack[x] for x in targets]
            solution = stack[-1]
            arguments.append(solution)
            stack.append(Function(primitive, arguments))
        elif name == "STRING":
            stack.append(Variable(var, List(STRING)))
            var += 1
            type_stack.append(List(STRING))
        elif name == "begin":
            stack.append(Primitive(name, REGEXP))
        elif name in name2type.keys():
            primitive = Primitive(name, name2type[name])
            stack.append(Function(primitive, [stack[-1]]))
    type_stack.append(stack[-1].type)
    type_request = FunctionType(*type_stack)
    return stack[-1], type_request


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert regexp original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "regexp.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON regexp file to be converted",
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        default=argument_default_values["output"],
        help=f"Output dataset file in ProgSynth format (default: '{argument_default_values['output']}')",
    )
    parsed_parameters = argument_parser.parse_args()
    convert_regexp(parsed_parameters.file, parsed_parameters.output)
