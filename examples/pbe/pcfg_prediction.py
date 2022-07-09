from typing import List
import atexit
import sys
import os
import random

import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from synth import Dataset, PBE, Task
from synth.nn import (
    BigramsPredictorLayer,
    Task2Tensor,
    print_model_summary,
)
from synth.pbe import IOEncoder
from synth.syntax import ConcreteCFG
from synth.utils import chrono

DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"


import argparse

parser = argparse.ArgumentParser(description="Evaluate model prediction")
parser.add_argument("--dataset", type=str, help="dataset (default: none)")
parser.add_argument(
    "--dsl",
    type=str,
    default=DEEPCODER,
    help="dsl (default: deepcoder)",
    choices=[DEEPCODER, DREAMCODER, REGEXP, CALCULATOR, TRANSDUCTION],
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="model.pt",
    help="output file (default: model.pt)",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    default=False,
    help="do not try to run things on cuda",
)
parser.add_argument(
    "--no-clean",
    action="store_true",
    default=False,
    help="do not delete intermediary model files",
)
gg = parser.add_argument_group("model parameters")
gg.add_argument(
    "-v",
    "--var-prob",
    type=float,
    default=0.2,
    help="variable probability (default: .2)",
)
gg.add_argument(
    "-ed",
    "--encoding-dimension",
    type=int,
    default=512,
    help="encoding dimension (default: 512)",
)
gg.add_argument(
    "-hd",
    "--hidden-size",
    type=int,
    default=512,
    help="hidden layer size (default: 512)",
)
g = parser.add_argument_group("training parameters")
g.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=16,
    help="batch size to compute PCFGs (default: 16)",
)
g.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=1,
    help="number of epochs (default: 1)",
)
g.add_argument(
    "--no-shuffle",
    action="store_true",
    default=False,
    help="do not shuffle dataset between epochs",
)
g.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=1e-3,
    help="learning rate (default: 1e-3)",
)
g.add_argument(
    "-wd",
    "--weight-decay",
    type=float,
    default=1e-4,
    help="weight decay (default: 1e-4)",
)
g.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")

parameters = parser.parse_args()
dataset_file: str = parameters.dataset
dsl_name: str = parameters.dsl
output_file: str = parameters.output
variable_probability: float = parameters.var_prob
batch_size: int = parameters.batch_size
epochs: int = parameters.epochs
lr: float = parameters.learning_rate
weight_decay: float = parameters.weight_decay
seed: int = parameters.seed
encoding_dimension: int = parameters.encoding_dimension
hidden_size: int = parameters.hidden_size
cpu_only: bool = parameters.cpu
no_clean: bool = parameters.no_clean
no_shuffle: bool = parameters.no_shuffle
should_generate_dataset: bool = False

random.seed(seed)
torch.manual_seed(seed)
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
upper_bound_type_size = 10
dsl_constant_types = set()
if dsl_name == DEEPCODER:
    from deepcoder.deepcoder import dsl, evaluator, lexicon

elif dsl_name == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, evaluator, lexicon

    max_list_length = 10

elif dsl_name == REGEXP:
    from regexp.regexp import dsl, evaluator, lexicon

    max_list_length = 10
elif dsl_name == CALCULATOR:
    from calculator.calculator import dsl, lexicon

    upper_bound_type_size = 5
elif dsl_name == TRANSDUCTION:
    from transduction.transduction import dsl, lexicon, constant_types

    upper_bound_type_size = 5
    dsl_constant_types = constant_types
else:
    print("Unknown dsl:", dsl_name, file=sys.stderr)
    sys.exit(1)
# ================================
# Load dataset & Task Generator
# ================================

if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)
# Load dataset

print(f"Loading {dataset_file}...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load(dataset_file)
    print("done in", c.elapsed_time(), "s")


# ================================
# Misc init
# ================================
# Get device
device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Logging
writer = SummaryWriter()
# ================================
# Neural Network creation
# ================================
# Generate the CFG dictionnary
all_type_requests = full_dataset.type_requests()
if all(task.solution is not None for task in full_dataset):
    max_depth = max(task.solution.depth() for task in full_dataset)
else:
    max_depth = 15  # TODO: set as parameter
cfgs = [
    ConcreteCFG.from_dsl(
        dsl,
        t,
        max_depth,
        upper_bound_type_size=upper_bound_type_size,
        constant_types=dsl_constant_types,
    )
    for t in all_type_requests
]
print(f"{len(all_type_requests)} type requests supported.")
print(f"Lexicon: [{min(lexicon)};{max(lexicon)}]")


class MyPredictor(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.bigram_layer = BigramsPredictorLayer(size, cfgs, variable_probability)
        encoder = IOEncoder(encoding_dimension, lexicon)
        self.packer = Task2Tensor(
            encoder, nn.Embedding(len(encoder.lexicon), size), size, device=device
        )
        self.rnn = nn.LSTM(size, size, 1)
        self.end = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
        )

    def forward(self, x: List[Task[PBE]]) -> Tensor:
        seq: PackedSequence = self.packer(x)
        _, (y, _) = self.rnn(seq)
        y: Tensor = y.squeeze(0)
        return self.bigram_layer(self.end(y))


predictor = MyPredictor(hidden_size).to(device)
print_model_summary(predictor)
optim = torch.optim.AdamW(predictor.parameters(), lr, weight_decay=weight_decay)

dataset_index = 0


@chrono.clock(prefix="train.do_batch")
def get_batch_of_tasks() -> List[Task[PBE]]:
    global dataset_index
    batch = full_dataset[dataset_index : dataset_index + batch_size]
    dataset_index += batch_size
    copy = []
    for task in batch:
        if task.solution is not None:
            copy.append(task)
    return copy


def do_batch(iter_number: int) -> None:
    batch = get_batch_of_tasks()
    batch_programs = [task.solution for task in batch]
    # Logging
    if all([p is not None for p in batch_programs]):
        writer.add_scalar(
            "program/depth", np.mean([p.depth() for p in batch_programs]), iter_number
        )
        mean_length = np.mean([p.length() for p in batch_programs])
        writer.add_scalar("program/length", mean_length, iter_number)
    with chrono.clock("train.do_batch.inference"):
        batch_outputs: Tensor = predictor(batch)
    # with chrono.clock("train.do_batch.tensor2pcfg"):
    #     batch_log_pcfg = [
    #         predictor.bigram_layer.tensor2pcfg(batch_outputs[i], task.type_request)
    #         for i, task in enumerate(batch)
    #     ]
    # Gradient descent
    with chrono.clock("train.do_batch.loss"):
        optim.zero_grad()
        with chrono.clock("train.do_batch.loss.compute"):
            loss = predictor.bigram_layer.loss_cross_entropy(
                batch_programs, batch_outputs
            )
            # loss = predictor.bigram_layer.loss_negative_log_prob(batch_programs, batch_log_pcfg)
        with chrono.clock("train.do_batch.loss.backprop"):
            loss.backward()
            optim.step()
    # Logging
    writer.add_scalar("train/loss", loss.item(), iter_number)
    # with torch.no_grad():
    #     loss = loss_negative_log_prob(
    #         batch_programs, batch_log_pcfg, length_normed=False
    #     )
    #     writer.add_scalar(
    #         "train/program_probability", np.exp(-loss.item()), iter_number
    #     )


def do_epoch(j: int) -> int:
    global dataset_index
    dataset_index = 0
    if not no_shuffle:
        random.shuffle(full_dataset.tasks)
    nb_batch_per_epoch = int(np.ceil(len(full_dataset) / batch_size))
    i = j
    for _ in tqdm.trange(nb_batch_per_epoch, desc="batchs"):
        do_batch(i)
        i += 1
    return i


def train() -> None:
    j = 0
    for ep in tqdm.trange(epochs, desc="epochs"):
        j = do_epoch(j)
        torch.save(predictor.state_dict(), f"{output_file}_epoch{ep}.tmp")


# Save on exit
def on_exit():
    writer.add_hparams(
        {
            "Learning rate": lr,
            "Weight Decay": weight_decay,
            "Batch Size": batch_size,
            "Epochs": epochs,
            "Variable Probability": variable_probability,
        },
        {},
    )
    writer.flush()
    writer.close()
    print(
        chrono.summary(
            time_formatter=lambda t: f"{int(t*1000)}ms" if not np.isnan(t) else "nan"
        )
    )


atexit.register(on_exit)


train()
torch.save(predictor.state_dict(), output_file)
if not no_clean:
    for ep in range(epochs):
        os.remove(f"{output_file}_epoch{ep}.tmp")
