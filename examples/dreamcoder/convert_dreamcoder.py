import json
from typing import Any, Dict, List, Callable


from synth.task import Task, Dataset
from synth.specification import PBE, Example
from synth.syntax.type_system import List


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = len([0 for t in tasks if t.solution])
    print(
        f"Converted {len(tasks)} tasks {int(100 * sols / len(tasks))}% containing solutions"
    )


def convert_dreamcoder(
    file: str,
    output_file: str = "dreamcoder.pickle",
) -> None:
    def load() -> Dataset[PBE]:
        tasks: List[Task[PBE]] = []
        with open(file, "rb") as fd:
            li: List[Dict[str, Any]] = json.load(fd)
            for task_dict in li:

                examples = [
                    Example([dico["i"]], dico["o"]) for dico in task_dict["examples"]
                ]
                spec = PBE(examples)
                tasks.append(
                    Task[PBE](
                        spec.guess_type(), spec, metadata={"name": task_dict["name"]}
                    )
                )
        return Dataset(tasks, metadata={"dataset": "dreamcoder", "source:": file})

    __convert__(load, output_file)


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert deepcoder original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "dreamcoder.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON file containing dreamcoder tasks to be converted",
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
    convert_dreamcoder(parsed_parameters.file, parsed_parameters.output)
