import argparse

import tqdm

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL

from synth import Dataset, PBE
from synth.utils import chrono
from synth.syntax import CFG

DREAMCODER = "dreamcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"


parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
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
parser.add_argument(
    "--no-unique",
    action="store_true",
    default=False,
    help="does not try to generate unique tasks",
)
parser.add_argument(
    "--constrained",
    action="store_true",
    default=False,
    help="tries to add constraints of the DSL to the grammar",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="verbose generation",
)
parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
output_file: str = parameters.output
seed: int = parameters.seed
max_depth: int = parameters.max_depth
gen_dataset_size: int = parameters.size
uniform: bool = parameters.uniform
no_unique: bool = parameters.no_unique
constrained: bool = parameters.constrained
verbose: bool = parameters.verbose
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon

if hasattr(dsl_module, "reproduce_dataset"):
    reproduce_dataset = dsl_module.reproduce_dataset
else:
    from synth.pbe.task_generator import reproduce_int_dataset as reproduce_dataset

constraints = []
if hasattr(dsl_module, "constraints") and constrained:
    constraints = dsl_module.constraints

if dsl_name == DREAMCODER:
    max_list_length = 10
# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
full_dataset: Dataset[PBE] = load_dataset(dsl_name, dataset_file)

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
        constraints=constraints,
        verbose=verbose,
    )
    cfgs = task_generator.type2pgrammar
    if constrained:
        cfgs = {
            t: CFG.depth_constraint(dsl, t, max_depth, min_variable_depth=0)
            for t in task_generator.type2pgrammar
        }
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
task_generator.uniques = not no_unique
task_generator.verbose = True
print("Generating dataset...", gen_dataset_size, end="", flush=True)
with chrono.clock("dataset.generate") as c:
    tasks = []
    pbar = tqdm.tqdm(total=gen_dataset_size, desc="tasks generated")
    for task in task_generator.generator():
        if constrained and task.solution not in cfgs[task.type_request]:
            continue
        pbar.update(1)
        tasks.append(task)
        if len(tasks) == gen_dataset_size:
            break
    pbar.close()
    gen_dataset = Dataset(
        tasks,
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
