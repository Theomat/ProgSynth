import json
from typing import Any, Callable, Dict, Tuple, List as TList

import tqdm

from synth import Task, Dataset, PBE, Example
from synth.syntax import (
    STRING,
    FunctionType,
    Type,
    Arrow,
    Function,
    Primitive,
    Program,
    Variable,
)
from examples.pbe.regexp.type_regex import REGEXP
from examples.pbe.transduction.transduction import dsl, evaluator, STREGEXP
from synth.syntax.dsl import DSL

name2type: Dict[str, Type] = {p.primitive: p.type for p in dsl.list_primitives}

def __decompose_dsl__(dsl: DSL, max_bound: int = 10):
    name2fulltype = {}
    dsl.instantiate_polymorphic_types(max_bound)
    for p in dsl.list_primitives:
        name = str(p)
        if isinstance(p.type, Arrow):
            name += ''.join([str(type) for type in p.type.arguments()])
        name2fulltype[name] = p.type
    return name2fulltype


name2fulltype = __decompose_dsl__(dsl, 5)


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = sum(1 for t in tasks if t.solution)
    print(f"Converted {len(tasks)} tasks {sols / len(tasks):.0%} containing solutions")
    # Integrity check
    for task in tqdm.tqdm(tasks, desc="integrity check"):
        for ex in task.specification.examples:
            assert evaluator.eval(task.solution, ex.inputs) == ex.output


def convert_transduction(
    file: str = "dataset/transduction_dataset.json",
    output_file: str = "transduction.pickle",
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

                prog, type_request = __transduction_str2prog(name)
                examples = [
                    Example([inp], out)
                    for inp, out in zip(inputs, outputs)
                    if out is not None
                ]
                if len(examples) < len(inputs):
                    continue
                tasks.append(
                    Task[PBE](type_request, PBE(examples), prog, {"name": name})
                )
        return Dataset(tasks, metadata={"dataset": "calculator", "source:": file})

    __convert__(load, output_file)


def __transduction_str2prog(s: str) -> Tuple[Program, Type]:
    parts = s.split("|")
    stack: TList[Program] = []
    var: int = 0
    type_stack: TList[Type] = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        if name == "STRING":
            stack.append(Variable(var, STRING))
            var += 1
            type_stack.append(STRING)
        elif name2type[name] == REGEXP:
            stack.append(Primitive(name, name2type[name]))
        else:
            targets = [int(x) for x in subparts]
            arguments = [stack[x] for x in targets]
            longname = name + ''.join([str(arg.type) for arg in arguments])
            primitive = Primitive(name, name2fulltype[longname])
            stack.append(Function(primitive, arguments))
    type_stack.append(stack[-1].type)
    type_request = FunctionType(*type_stack)
    return stack[-1], type_request


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert transduction original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "transduction.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON transduction file to be converted",
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
    convert_transduction(parsed_parameters.file, parsed_parameters.output)
