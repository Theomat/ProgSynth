from deepcoder import dsl, evaluator, lexicon
from typing import List


import tqdm

import torch
from torch import Tensor
import torch.nn as nn

import numpy as np

from synth import Dataset, Task
from synth.nn import (
    PrimitivePredictorLayer,
    print_model_summary,
)
from synth.specification import NLP
from synth.nlp import NLPEncoder
from synth.syntax import ConcreteCFG, Function, Variable, Arrow
from synth.utils import chrono

seed: int = 1
cpu_only = True
batch_size = 2

torch.manual_seed(seed)
# =============================
# Misc init
# ================================
# Get device
device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset
dataset = []
for P in dsl.list_primitives:
    if "[" in P.primitive:
        continue
    if isinstance(P.type, Arrow) and len(P.type.arguments()) == 1:
        dataset.append(
            Task[NLP](
                P.type,
                NLP(P.primitive.lower() + " the list `list`."),
                Function(P, [Variable(0, P.type.arguments()[0])]),
            )
        )


full_dataset = Dataset(dataset)
# ================================
# Neural Network creation
# ================================
# Generate the CFG dictionnary
all_type_requests = full_dataset.type_requests()
print(f"{len(all_type_requests)} type requests supported.")
print(f"Lexicon: [{min(lexicon)};{max(lexicon)}]")


class MyPredictor(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.primitive_layer = PrimitivePredictorLayer(size, dsl, 0.2)
        self.encoder = NLPEncoder()
        input_size = self.encoder.embedding_size
        self.rnn = nn.LSTM(input_size, size, 1, batch_first=True)

        self.end = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
        )

    def forward(self, x: List[Task[NLP]]) -> Tensor:
        xx = [self.encoder.encode(task) for task in x]
        xxx = torch.stack(xx).squeeze(1)
        y0, _ = self.rnn(xxx)
        y = y0.data[:, -1, :]
        return self.primitive_layer(self.end(y))


predictor = MyPredictor(2000).to(device)
print_model_summary(predictor)
optim = torch.optim.AdamW(predictor.parameters(), 1e-3)

dataset_index = 0


@chrono.clock(prefix="train.do_batch")
def get_batch_of_tasks() -> List[Task[NLP]]:
    global dataset_index
    batch = full_dataset[dataset_index : dataset_index + batch_size]
    dataset_index += batch_size
    return batch


def do_batch(iter_number: int) -> None:
    nb_batch_per_epoch = int(np.ceil(len(full_dataset) / batch_size))

    batch = get_batch_of_tasks()
    batch_programs = [task.solution for task in batch]
    # Logging
    with chrono.clock("train.do_batch.inference"):
        batch_outputs: Tensor = predictor(batch)
    # Gradient descent
    with chrono.clock("train.do_batch.loss"):
        optim.zero_grad()
        with chrono.clock("train.do_batch.loss.compute"):
            loss = predictor.primitive_layer.loss(batch_programs, batch_outputs)
        with chrono.clock("train.do_batch.loss.backprop"):
            loss.backward()
            optim.step()
        if (iter_number + 1) % nb_batch_per_epoch == 0:
            print("Loss=", loss.item())
            task = batch[0]
            print(task)
            out = batch_outputs[0]
            pcfg = predictor.primitive_layer.tensor2pcfg(
                out, task.type_request, max_depth=2
            ).to_pcfg()
            print(pcfg)


def do_epoch(j: int) -> int:
    global dataset_index
    dataset_index = 0
    nb_batch_per_epoch = int(np.ceil(len(full_dataset) / batch_size))
    i = j
    for _ in tqdm.trange(nb_batch_per_epoch, desc="batchs"):
        do_batch(i)
        i += 1
    return i


def train() -> None:
    j = 0
    for ep in tqdm.trange(100, desc="epochs"):
        j = do_epoch(j)


train()
