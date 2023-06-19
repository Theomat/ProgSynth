from synth import Dataset, PBE
from synth.utils import chrono
from synth.library import learn, make_score_probabilistic, score_description

from dsl_loader import add_dsl_choice_arg, load_DSL
from dataset_loader import add_dataset_choice_arg, load_dataset

import argparse

parser = argparse.ArgumentParser(description="Learn a new primitive based on a dataset")
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "--probabilistic",
    action="store_true",
    help="Maximise probability instead of reducing description size",
)
parameters = parser.parse_args()
dsl_name: str = parameters.dsl
proba: bool = parameters.probabilistic
dataset_file: str = parameters.dataset
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

programs = [t.solution for t in full_dataset if t.solution is not None]
score_fn = make_score_probabilistic(programs, False) if proba else score_description
score, prog = learn(programs, score_fn, progress=True)
if proba:
    print(f"Best program is {prog} which bumps the programs set's log prob to:{score}.")
else:
    print(f"Best program is {prog} which reduces description size by:{score}.")
# print(f"This would reduce the size by {(size - 1) * occs}.")
# score, prog = learn(
#     [t.solution for t in full_dataset if t.solution is not None], progress=True
# )
# # print(f"Found {occs} occurences of {prog}.")
# print(f"This would reduce the size by {score}.")
