from typing import List
import atexit
import sys

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
    loss_negative_log_prob,
    Task2Tensor,
)
from synth.pbe import reproduce_dataset, IOEncoder
from synth.syntax import ConcreteCFG
from synth.utils import chrono, gen_take

DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"


import argparse

parser = argparse.ArgumentParser(description="Evaluate model prediction")
parser.add_argument(
    "-d", "--dataset", type=str, default=DEEPCODER, help="dataset (default: deepcoder)"
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
# g.add_argument(
#     "-e",
#     "--epochs",
#     type=int,
#     default=1,
#     help="number of epochs (default: 1)",
# )
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
g.add_argument(
    "--size", type=int, default=1000, help="generated datset size (default: 1000)"
)

parameters = parser.parse_args()
dataset: str = parameters.dataset
output_file: str = parameters.output
variable_probability: float = parameters.var_prob
batch_size: int = parameters.batch_size
epochs: int = 1  # parameters.epochs
lr: float = parameters.learning_rate
weight_decay: float = parameters.weight_decay
seed: int = parameters.seed
encoding_dimension: int = parameters.encoding_dimension
hidden_size: int = parameters.hidden_size
gen_dataset_size: int = parameters.size
cpu_only: bool = parameters.cpu

torch.manual_seed(seed)
# ================================
# Load constants specific to dataset
# ================================
dataset_file = f"{dataset}.pickle"
max_list_length = None
if dataset == DEEPCODER:
    from deepcoder.deepcoder import dsl, evaluator

elif dataset == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, evaluator

    max_list_length = 10
else:
    print("Unknown dataset:", dataset, file=sys.stderr)
    sys.exit(0)
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
        full_dataset, dsl, evaluator, seed, max_list_length=max_list_length
    )
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
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
all_type_requests = set(task_generator.type2pcfg.keys())
if dataset == DEEPCODER:
    max_depth = max(task.solution.depth() for task in full_dataset)
elif dataset == DREAMCODER:
    max_depth = 5
cfgs = [ConcreteCFG.from_dsl(dsl, t, max_depth) for t in all_type_requests]
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
        )

    def forward(self, x: List[Task[PBE]]) -> Tensor:
        seq: PackedSequence = self.packer(x)
        y0, _ = self.rnn(seq)
        y = y0.data
        return self.bigram_layer(self.end(y))


predictor = MyPredictor(hidden_size).to(device)
optim = torch.optim.AdamW(predictor.parameters(), lr, weight_decay=weight_decay)


@chrono.clock(prefix="train.do_batch")
def get_batch_of_tasks() -> List[Task[PBE]]:
    return gen_take(task_generator.generator(), batch_size)


def do_batch(iter_number: int) -> None:
    batch = get_batch_of_tasks()
    batch_programs = [task.solution for task in batch]
    # Logging
    writer.add_scalar(
        "program/depth", np.mean([p.depth() for p in batch_programs]), iter_number
    )
    mean_length = np.mean([p.length() for p in batch_programs])
    writer.add_scalar("program/length", mean_length, iter_number)
    with chrono.clock("train.do_batch.inference"):
        batch_outputs: Tensor = predictor(batch)
    with chrono.clock("train.do_batch.tensor2pcfg"):
        batch_log_pcfg = [
            predictor.bigram_layer.tensor2pcfg(batch_outputs[i], task.type_request)
            for i, task in enumerate(batch)
        ]
    # Gradient descent
    with chrono.clock("train.do_batch.loss"):
        optim.zero_grad()
        with chrono.clock("train.do_batch.loss.compute"):
            loss = loss_negative_log_prob(batch_programs, batch_log_pcfg)
        with chrono.clock("train.do_batch.loss.backprop"):
            loss.backward()
            optim.step()
    # Logging
    writer.add_scalar("loss/train", loss.item(), iter_number)

    total = sum(task_generator.generated_types.values())
    for t, v in task_generator.difficulty.items():
        writer.add_scalar(
            f"input sampling success/{t}", v[1] / max(1, (v[0] + v[1])), iter_number
        )
        writer.add_scalar(
            f"task type distribution/{t}",
            task_generator.generated_types[t] / total,
            iter_number,
        )


def do_epoch(j: int) -> int:
    nb_batch_per_epoch = int(gen_dataset_size / batch_size + 0.5)
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
