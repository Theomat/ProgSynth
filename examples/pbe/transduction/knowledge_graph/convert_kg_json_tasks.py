import json

from synth.specification import PBE, Example, TaskSpecification
from synth.task import Dataset, Task


def convert(dataset_file: str, output_file: str):
    tasks = []

    with open(dataset_file) as fd:
        lst = json.load(fd)
        for el in lst:
            spec = PBE([Example(ex["inputs"], ex["output"]) for ex in el["examples"]])
            task = Task[PBE](
                spec.guess_type(), specification=spec, metadata=el["metadata"]
            )
            tasks.append(task)

    dataset: Dataset[TaskSpecification] = Dataset(tasks)
    dataset.save(output_file)


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert knowledge graph JSON dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "constants.pickle",
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
    convert(parsed_parameters.file, parsed_parameters.output)
