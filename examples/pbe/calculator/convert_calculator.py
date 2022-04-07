import json
from typing import Any, Callable, Dict, Tuple, List as TList

import tqdm

from synth import Task, Dataset, PBE, Example
from synth.syntax import (
    INT,
    FunctionType,
    List,
    Type,
    Arrow,
    Function,
    Primitive,
    Program,
    Variable,
    PrimitiveType,
    PolymorphicType,
)

from calculator import dsl, evaluator, FLOAT

# this dictionary contains the primitives as defined in the dsl
name2type: Dict[str, Type] = {p.primitive: p.type for p in dsl.list_primitives}
# this dictionary contains the instantiated primitives, that is to say after removal of polymorphic type
name2fulltype = {}
dsl.instantiate_polymorphic_types(5)
for p in dsl.list_primitives:
    if isinstance(p.type, Arrow):
        name = str(p) + str(p.type.arguments()[0])
    else:
        name = p.primitive
    name2fulltype[name] = p.type


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = sum(1 for t in tasks if t.solution)
    print(f"Converted {len(tasks)} tasks {sols / len(tasks):.0%} containing solutions")
    # Integrity check
    for task in tqdm.tqdm(tasks, desc="integrity check"):
        for ex in task.specification.examples:
            assert evaluator.eval(task.solution, ex.inputs) == ex.output


def convert_calculator(
    file: str = "dataset/calculator_dataset.json",
    output_file: str = "calculator.pickle",
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

                prog, type_request = __calculator_str2prog(name)
                examples = [
                    Example(inp, out)
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


"""
Parser of program stored in the json file, for a given task.
s: program attribute represented as a string
returns: Tuple[Program, Type] where program is the solution of type Type to the task
"""


def __calculator_str2prog(s: str) -> Tuple[Program, Type]:
    parts = s.split("|")
    stack: TList[Program] = []
    var: int = 0
    type_stack: TList[Type] = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        # possible inputs, int or float in our case
        if name == "INT":
            stack.append(Variable(var, INT))
            var += 1
            type_stack.append(INT)
            continue
        if name == "FLOAT":
            stack.append(Variable(var, FLOAT))
            var += 1
            type_stack.append(FLOAT)
            continue
        # primitives that serve as constants
        if name in ["1", "2", "3"]:
            primitive = Primitive(name, name2fulltype[name])
            stack.append(primitive)
        else:  # other primitives are functions, we want to add their type
            targets = [int(x) for x in subparts]
            arguments = [stack[x] for x in targets]
            longname = name + str(arguments[-1].type)
            primitive = Primitive(name, name2fulltype[longname])
            stack.append(Function(primitive, arguments))
    type_stack.append(stack[-1].type)
    type_request = FunctionType(*type_stack)
    return stack[-1], type_request


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert calculator original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "calculator.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON calculator file to be converted",
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
    convert_calculator(parsed_parameters.file, parsed_parameters.output)
