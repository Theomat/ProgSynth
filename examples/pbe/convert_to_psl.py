import argparse
import os.path as path
from typing import Union

from dsl_loader import add_dsl_choice_arg, load_DSL
from dataset_loader import add_dataset_choice_arg, load_dataset

from synth import Dataset, PBE
from synth.specification import PBEWithConstants


parser = argparse.ArgumentParser(
    description="Convert a ProgSynth dataset to the PSL format"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "dest_folder",
    type=str,
    help="destination folder",
)
parser.add_argument(
    "logics",
    type=str,
    help="logics used",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
dest_folder: str = parameters.dest_folder
logics: str = parameters.logics
# ================================
# Load constants specific to DSL
# ================================
dsl_module = load_DSL(dsl_name)
dsl = dsl_module.dsl
# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
full_dataset: Dataset[Union[PBE, PBEWithConstants]] = load_dataset(
    dsl_name, dataset_file
)
COMMENT_PREFIX = "#"


for i, task in enumerate(full_dataset.tasks):
    spec = task.specification
    name = task.metadata.get("name", f"{dsl_name}_{i}")
    filepath = path.join(dest_folder, name + ".psl")
    try:
        fd = open(filepath, "w")
        fd.close()
    except:
        filepath = path.join(dest_folder, f"{dsl_name}_{i}" + ".psl")

    with open(filepath, "w") as fd:
        fd.write(f"(set-logic {logics})\n")

        fd.write(f"\n{COMMENT_PREFIX} Function Synthesis\n")
        fd.write("(synth-fun f ")
        for j, arg in enumerate(task.type_request.arguments()):
            fd.write(f"(x{j+1} {arg}) ")
        fd.write(f"{task.type_request.returns()})\n")

        fd.write(f"\n{COMMENT_PREFIX} PBE Examples\n")
        for example in spec.examples:
            inputs = " ".join(map(str, example.inputs))
            output = str(example.output)
            fd.write(f"(constraint-pbe (f {inputs}) {output})\n")

        if isinstance(spec, PBEWithConstants):
            constants = spec.constants
            fd.write(f"\n{COMMENT_PREFIX} Constants\n")
            for type, values in spec.constants.items():
                allowed = " ".join(map(str, values))
                fd.write(f"(define-const {type} {allowed})\n")

        fd.write("\n(check-progsynth)\n")
        if task.solution is not None:
            fd.write(f"\n(solution-pbe {task.solution})\n")
        lines = []
        for name, val in task.metadata.items():
            if name == "name":
                continue
            lines.append(f"{COMMENT_PREFIX} {name}: {val}")
        if lines:
            fd.write(f"\n{COMMENT_PREFIX} Metadata:\n")
            fd.writelines(lines)
