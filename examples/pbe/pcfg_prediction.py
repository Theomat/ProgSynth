from typing import List
import atexit

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

# ================================
# Change dataset
# ================================
from deepcoder.deepcoder import dsl, evaluator
uniform_pcfg = False

from dreamcoder.dreamcoder import dsl, evaluator
uniform_pcfg = True
# ================================
# Tunable parameters
# ================================
# Model parameters
variable_probability = 0.2

# Training parameters
epochs = 2
batch_size = 16

lr = 1e-3
weight_decay = 1e-4
# ================================
# Initialisation
# ================================
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Load dataset
print("Loading dataset...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load("./deepcoder.pickle")
    print("done in", c.elapsed_time(), "s")
# Reproduce dataset distribution
print("Reproducing dataset...", end="")
with chrono.clock("dataset.reproduce") as c:
    task_generator, lexicon = reproduce_dataset(
        full_dataset, dsl, evaluator, 0, uniform_pcfg=uniform_pcfg
    )
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
# Generate the CFG dictionnary
all_type_requests = set(task_generator.type2pcfg.keys())
max_depth = max(task.solution.depth() for task in full_dataset)
cfgs = [ConcreteCFG.from_dsl(dsl, t, max_depth) for t in all_type_requests]
# Logging
writer = SummaryWriter()
# ================================
# Neural Network creation
# ================================


class MyPredictor(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.bigram_layer = BigramsPredictorLayer(size, dsl, cfgs, variable_probability)
        encoder = IOEncoder(512, lexicon)
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


predictor = MyPredictor(512).to(device)
print("Model:", predictor)
optim = torch.optim.Adam(predictor.parameters(), lr, weight_decay=weight_decay)


@chrono.clock(prefix="train")
def do_batch(iter_number: int) -> None:
    with chrono.clock("train.do_batch.task_generation"):
        batch = gen_take(task_generator.generator(), batch_size)
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
    writer.add_scalar(
        "length normed probability/train",
        np.exp(-loss.item()) / mean_length,
        iter_number,
    )

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
    nb_batch_per_epoch = int(len(full_dataset) / batch_size + 0.5)
    i = j
    for _ in tqdm.trange(nb_batch_per_epoch, desc="batchs"):
        do_batch(i)
        i += 1
    return i


def train() -> None:
    j = 0
    for _ in tqdm.trange(epochs, desc="epochs"):
        j = do_epoch(j)


def test() -> None:
    with torch.no_grad():
        taken = 0
        tasks = full_dataset.tasks
        total_loss = None
        iter_number = 0
        while taken < len(tasks):
            batch_end = min(taken + batch_size, len(tasks))
            batch = tasks[taken:batch_end]

            batch_programs = [task.solution for task in batch]
            batch_outputs: Tensor = predictor(batch)

            batch_log_pcfg = [
                predictor.bigram_layer.tensor2pcfg(batch_outputs[i], task.type_request)
                for i, task in enumerate(batch)
            ]
            loss = loss_negative_log_prob(batch_programs, batch_log_pcfg, torch.sum)
            total_loss = total_loss + loss if total_loss else loss
            # Logging
            iter_number += 1
            writer.add_scalar(
                "loss/test", loss.item() / (batch_end - taken + 1), iter_number
            )
            mean_length = np.mean([p.length() for p in batch_programs])
            writer.add_scalar(
                "length normed probability/test",
                np.exp(-loss.item()) / mean_length,
                iter_number,
            )
            # Update taken
            taken = batch_end
        total_loss /= taken
        print("Test loss:", total_loss.item())


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
test()
