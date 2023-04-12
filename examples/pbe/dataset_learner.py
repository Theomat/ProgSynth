from synth import Dataset, PBE
from synth.utils import chrono
from synth.library import learn

from dsl_loader import add_dsl_choice_arg, load_DSL

import argparse

parser = argparse.ArgumentParser(description="Learn a new primitive based on a dataset")
add_dsl_choice_arg(parser)
parser.add_argument(
    "--dataset",
    type=str,
    default="{dsl_name}.pickle",
    help="dataset file (default: {dsl_name}.pickle)",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset.format(dsl_name=dsl_name)
# ================================
# Load constants specific to DSL
# ================================
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon
# ================================
# Load dataset
# ================================
# Load dataset
print(f"Loading {dataset_file}...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load(dataset_file)
    print("done in", c.elapsed_time(), "s")

size, occs, prog = learn(
    [t.solution for t in full_dataset if t.solution is not None], progress=True
)
print(f"Found {occs} occurences of {prog}.")
print(f"This would reduce the size by {(size - 1) * occs}.")
