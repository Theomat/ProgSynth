import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from model_loader import (
    add_model_choice_arg,
    instantiate_predictor,
)


from synth.nn import print_model_summary
from synth.syntax import CFG, UCFG
from synth.pruning.constraints import add_dfta_constraints
from synth.pbe.io_encoder import IOEncoder


import argparse


parser = argparse.ArgumentParser(description="Visualize model")
parser.add_argument("-m", "--model", default="", type=str, help="model file")
add_dataset_choice_arg(parser)
add_dsl_choice_arg(parser)
add_model_choice_arg(parser)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
cpu_only: bool = parameters.cpu
model_file: str = parameters.model
constrained: bool = parameters.constrained
# Get device
device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Load DSL ================================================================
dsl_module = load_DSL(dsl_name)
dsl, lexicon = dsl_module.dsl, dsl_module.lexicon
constraints = []
if constrained and hasattr(dsl_module, "constraints"):
    constraints = dsl_module.constraints
# Load Dataset ============================================================
full_dataset = load_dataset(dsl_name, dataset_file)
# Load CFGs ===============================================================
all_type_requests = full_dataset.type_requests()

if all(task.solution is not None for task in full_dataset):
    max_depth = max(task.solution.depth() for task in full_dataset)
else:
    max_depth = 5  # TODO: set as parameter
cfgs = [
    CFG.depth_constraint(
        dsl,
        t,
        max_depth,
        upper_bound_type_size=10,
        constant_types=set(),
        min_variable_depth=0,
    )
    for t in all_type_requests
]
cfgs = [
    UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, constraints, progress=False), 2
    )
    if constrained
    else cfg
    for cfg in cfgs
]

writer = SummaryWriter(comment=f"model_vizualizer_{model_file}")
# Load Model ==============================================================
predictor = instantiate_predictor(parameters, cfgs, lexicon)
predictor.load_state_dict(torch.load(model_file, map_location=device))
predictor = predictor.to(device)
predictor.eval()
print_model_summary(predictor)
# Plot embeddings =========================================================
print("Generating embeddings data:")
encoder = predictor.packer.encoder
embedder = predictor.packer.embedder
# For now this part assumes isinstance(encoder, IOEncoder)
assert isinstance(encoder, IOEncoder)
encoded = []
for l in lexicon:
    encoder.__encode_element__(l, encoded)
# Built as a Tensor
res = torch.LongTensor(encoded).to(device).reshape((-1, 1))
output: Tensor = embedder(res).squeeze()
writer.add_embedding(output, metadata=lexicon)
# END ====================================================================
print("Additional model data can now be viewed with TensorBoard!")
writer.close()
