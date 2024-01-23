import argparse
import atexit
import os
import sys
from typing import List, Optional, Set, Tuple, Union

import tqdm

import torch

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from model_loader import (
    add_model_choice_arg,
    instantiate_predictor,
)


from synth import Dataset, PBE
from synth.filter import add_dfta_constraints
from synth.syntax import CFG, UCFG, ProbDetGrammar, ProbUGrammar, DSL, Type
from synth.utils import load_object, save_object


parser = argparse.ArgumentParser(
    description="Predict Probabilistic grammars for a dataset"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument("-m", "--model", default="", type=str, help="model file")
add_model_choice_arg(parser)
parser.add_argument(
    "--support",
    type=str,
    default=None,
    help="train dataset to get the set of supported type requests",
)
g = parser.add_argument_group("pcfg prediction parameter")
g.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=16,
    help="batch size to compute PCFGs (default: 16)",
)


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
model_file: str = parameters.model
batch_size: int = parameters.batch_size
constrained: bool = parameters.constrained
max_depth: int = parameters.max_depth
ngram: int = parameters.ngram
support: Optional[str] = (
    None if not parameters.support else parameters.support.format(dsl_name=dsl_name)
)


if not os.path.exists(model_file) or not os.path.isfile(model_file):
    print("Model must be a valid model file!", file=sys.stderr)
    sys.exit(1)
elif not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)
elif support is not None and (
    not os.path.exists(support) or not os.path.isfile(support)
):
    print("Support dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)


start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

supported_type_requests = Dataset.load(support).type_requests() if support else None

# ================================
# Load constants specific to dataset
# ================================


def load_dsl_and_dataset() -> Tuple[
    Dataset[PBE], DSL, List[int], str, List[str], Set[Type]
]:
    dsl_module = load_DSL(dsl_name)
    dsl, lexicon = dsl_module.dsl, dsl_module.lexicon
    constant_types: Set[Type] = set()
    constraints = []
    if constrained and hasattr(dsl_module, "constraints"):
        constraints = dsl_module.constraints
    if hasattr(dsl_module, "constant_types"):
        constant_types = dsl_module.constant_types
    # ================================
    # Load dataset
    # ================================
    full_dataset = load_dataset(dsl_name, dataset_file)

    start_index = (
        0
        if not os.path.sep in model_file
        else (len(model_file) - model_file[::-1].index(os.path.sep))
    )
    model_name = model_file[start_index : model_file.index(".", start_index)]
    return full_dataset, dsl, lexicon, model_name, constraints, constant_types


# Produce PCFGS ==========================================================
@torch.no_grad()
def produce_pcfgs(
    full_dataset: Dataset[PBE],
    dsl: DSL,
    lexicon: List[int],
    constraints: List[str],
    constant_types: Set[Type],
) -> Union[List[ProbDetGrammar], List[ProbUGrammar]]:
    # ================================
    # Load already done PCFGs
    # ================================
    dir = os.path.realpath(os.path.dirname(model_file))
    start_index = (
        0
        if not os.path.sep in model_file
        else (len(model_file) - model_file[::-1].index(os.path.sep))
    )
    model_name = model_file[start_index : model_file.index(".", start_index)]
    file = os.path.join(dir, f"pcfgs_{dataset_name}_{model_name}.pickle")
    pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]] = []
    if os.path.exists(file):
        pcfgs = load_object(file)
    tasks = full_dataset.tasks
    tasks = [
        t
        for t in tasks
        if supported_type_requests is None or t.type_request in supported_type_requests
    ]
    done = len(pcfgs)
    # ================================
    # Skip if possible
    # ================================
    if done >= len(tasks):
        return pcfgs

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # ================================
    # Neural Network creation
    # ================================
    # Generate the CFG dictionnary
    all_type_requests = (
        full_dataset.type_requests() if support is None else supported_type_requests
    )

    cfgs = [
        CFG.depth_constraint(
            dsl,
            t,
            max_depth,
            constant_types=constant_types,
            min_variable_depth=0,
            n_gram=ngram,
        )
        for t in all_type_requests
    ]
    cfgs = [
        UCFG.from_DFTA_with_ngrams(
            add_dfta_constraints(cfg, constraints, progress=False), ngram
        )
        if constrained
        else cfg
        for cfg in cfgs
    ]

    predictor = instantiate_predictor(parameters, cfgs, lexicon)
    predictor.load_state_dict(torch.load(model_file, map_location=device))
    predictor = predictor.to(device)
    predictor.eval()
    # ================================
    # Predict PCFG
    # ================================
    def save_pcfgs() -> None:
        print("Saving PCFGs...", end="")
        save_object(file, pcfgs)
        print("done!")

    atexit.register(save_pcfgs)

    pbar = tqdm.tqdm(total=len(tasks) - done, desc="PCFG prediction")
    while done < len(tasks):
        end = min(len(tasks), done + batch_size)
        batch = tasks[done:end]
        pbar.update(end - done)
        done = end
        batch_outputs = predictor(batch)

        for task, tensor in zip(batch, batch_outputs):
            obj = predictor.bigram_layer.tensor2log_prob_grammar(
                tensor, task.type_request
            )
            pcfgs.append(
                obj.to_prob_u_grammar() if constrained else obj.to_prob_det_grammar()
            )
    pbar.close()
    save_pcfgs()
    atexit.unregister(save_pcfgs)

    return pcfgs


# Main ====================================================================

if __name__ == "__main__":
    (
        full_dataset,
        dsl,
        lexicon,
        model_name,
        constraints,
        constant_types,
    ) = load_dsl_and_dataset()

    pcfgs = produce_pcfgs(full_dataset, dsl, lexicon, constraints, constant_types)
