import sys

from synth import Dataset, PBE
from synth.utils import chrono, gen_take

DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"
CALCULATOR = "calculator"


import argparse

parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
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
    "-o",
    "--output",
    type=str,
    default="dataset.pickle",
    help="output file (default: dataset.pickle)",
)
parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
parser.add_argument(
    "--size", type=int, default=1000, help="generated dataset size (default: 1000)"
)
parser.add_argument(
    "--max-depth", type=int, default=5, help="solutions max depth (default: 5)"
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset.format(dsl_name=dsl_name)
output_file: str = parameters.output
seed: int = parameters.seed
max_depth: int = parameters.max_depth
gen_dataset_size: int = parameters.size
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
if dsl_name == DEEPCODER:
    from synth.pbe import reproduce_dataset
    from deepcoder.deepcoder import dsl, evaluator, lexicon

elif dsl_name == DREAMCODER:
    from synth.pbe import reproduce_dataset
    from dreamcoder.dreamcoder import dsl, evaluator, lexicon

    max_list_length = 10
elif dsl_name == CALCULATOR:
    from calculator.calculator_task_generator import reproduce_dataset
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
# Reproduce dataset distribution
print("Reproducing dataset...", end="", flush=True)
with chrono.clock("dataset.reproduce") as c:
    task_generator, lexicon = reproduce_dataset(
        full_dataset,
        dsl,
        evaluator,
        seed,
        max_list_length=max_list_length,
        default_max_depth=max_depth,
    )
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
print("Generating dataset...", end="", flush=True)
with chrono.clock("dataset.generate") as c:
    gen_dataset = Dataset(
        gen_take(task_generator.generator(), gen_dataset_size),
        {
            "seed": seed,
            "max_depth": max_depth,
            "dsl": dsl_name,
            "max_list_length": max_list_length,
        },
    )
    print("done in", c.elapsed_time(), "s")
print("Saving dataset...", end="", flush=True)
with chrono.clock("dataset.save") as c:
    gen_dataset.save(output_file)
    print("done in", c.elapsed_time(), "s")

# ================================
# Print some stats
# ================================
# Generate the CFG dictionnary
all_type_requests = gen_dataset.type_requests()
print(f"{len(all_type_requests)} type requests supported.")
print(f"Lexicon: [{min(lexicon)};{max(lexicon)}]")
