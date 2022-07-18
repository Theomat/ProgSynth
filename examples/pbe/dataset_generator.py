import argparse

from dsl_loader import add_dsl_choice_arg, load_DSL

from synth import Dataset, PBE
from synth.utils import chrono, gen_take

DREAMCODER = "dreamcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"


parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
)
add_dsl_choice_arg(parser)

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
parser.add_argument(
    "--uniform", action="store_true", default=False, help="use uniform PCFGs"
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset.format(dsl_name=dsl_name)
output_file: str = parameters.output
seed: int = parameters.seed
max_depth: int = parameters.max_depth
gen_dataset_size: int = parameters.size
uniform: bool = parameters.uniform
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon

if dsl_name == DREAMCODER:
    from synth.pbe import reproduce_dataset

    max_list_length = 10
elif dsl_name == REGEXP:
    from regexp.task_generator_regexp import reproduce_dataset
elif dsl_name == CALCULATOR:
    from calculator.calculator_task_generator import reproduce_dataset
elif dsl_name == TRANSDUCTION:
    from transduction.transduction_task_generator import reproduce_dataset
else:
    from synth.pbe import reproduce_dataset

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
        uniform_pgrammar=uniform,
    )
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
task_generator.uniques = True
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
