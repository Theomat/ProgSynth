import os
from typing import Callable, List, Optional, Tuple
import csv
import pickle

import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

import numpy as np

from synth import Dataset, PBE, Task
from synth.nn import (
    BigramsPredictorLayer,
    Task2Tensor,
)
from synth.pbe import IOEncoder
from synth.syntax import ConcreteCFG, ConcretePCFG, enumerate_pcfg
from synth.syntax.program import Program
from synth.utils import chrono

# ================================
# Change dataset
# ================================
DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"

dataset = DEEPCODER
# ================================
# Tunable parameters
# ================================
model_folder = "."
output_folder = "."
# Model parameters
variable_probability = 0.2

# Memory parameters
batch_size = 16

# In seconds
task_timeout = 300
# ================================
# Load constants specific to dataset
# ================================
dataset_file = f"{dataset}.pickle"

if dataset == DEEPCODER:
    from deepcoder.deepcoder import dsl, evaluator, lexicon

    uniform_pcfg = False
elif dataset == DREAMCODER:
    from dreamcoder.dreamcoder import dsl, evaluator

    lexicon = []  # TODO


# ================================
# Load dataset
# ================================
# Load dataset
print(f"Loading {dataset_file}...", end="")
with chrono.clock("dataset.load") as c:
    full_dataset: Dataset[PBE] = Dataset.load(dataset_file)
    print("done in", c.elapsed_time(), "s")


# Produce PCFGS ==========================================================
@torch.no_grad()
def produce_pcfgs() -> List[ConcreteCFG]:
    # ================================
    # Load already done PCFGs
    # ================================
    file = os.path.join(model_folder, f"{dataset}_pcfgs.pickle")
    pcfgs: List[ConcretePCFG] = []
    if os.path.exists(file):
        with open(file, "rb") as fd:
            pcfgs = pickle.load(fd)
    tasks = full_dataset.tasks
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
    all_type_requests = full_dataset.type_requests()
    if dataset == DEEPCODER:
        max_depth = max(task.solution.depth() for task in full_dataset)
    elif dataset == DREAMCODER:
        max_depth = 5
    cfgs = [ConcreteCFG.from_dsl(dsl, t, max_depth) for t in all_type_requests]

    class MyPredictor(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.bigram_layer = BigramsPredictorLayer(
                size, dsl, cfgs, variable_probability
            )
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

    predictor = MyPredictor(512)
    predictor.load_state_dict(
        torch.load(os.path.join(model_folder, f"{dataset}_model.pt"))
    )
    predictor = predictor.to(device)
    predictor.eval()
    # ================================
    # Predict PCFG
    # ================================

    pbar = tqdm.tqdm(total=len(tasks) - done, desc="PCFG prediction")
    while done < len(tasks):
        end = min(len(tasks), done + batch_size)
        batch = tasks[done:end]
        pbar.update(end - done)
        done = end
        batch_outputs = predictor(batch)

        for task, tensor in zip(batch, batch_outputs):
            pcfgs.append(
                predictor.bigram_layer.tensor2pcfg(tensor, task.type_request).to_pcfg()
            )
    pbar.close()
    with open(file, "wb") as fd:
        pickle.dump(pcfgs, fd)
    return pcfgs


# Enumeration methods =====================================================
def enumerative_search(
    pcfgs: List[ConcretePCFG],
    trace: List[Tuple[bool, float]],
    method: Callable[
        [Task[PBE], ConcretePCFG], Tuple[bool, float, int, Optional[Program]]
    ],
) -> None:
    start = len(trace)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks")
    for task, pcfg in zip(full_dataset.tasks[start:], pcfgs[start:]):
        trace.append(method(task, pcfg))
        pbar.update(1)
        evaluator.clear_cache()
    pbar.close()


def base(
    task: Task[PBE], pcfg: ConcretePCFG
) -> Tuple[bool, float, int, Optional[Program]]:
    time = 0.0
    programs = 0
    with chrono.clock("search.base") as c:
        for program in enumerate_pcfg(pcfg):
            time = c.elapsed_time()
            if time >= task_timeout:
                return (False, time, programs, None)
            programs += 1
            failed = False
            for ex in task.specification.examples:
                if evaluator.eval(program, ex.inputs) != ex.output:
                    failed = True
                    break
            if not failed:
                return (True, c.elapsed_time(), programs, program)
    return (False, time, programs, None)


# Main ====================================================================

if __name__ == "__main__":
    methods = [
        ("base", base),
    ]

    pcfgs = produce_pcfgs()
    should_exit = False

    for name, method in methods:
        file = os.path.join(output_folder, f"{dataset}_{name}.csv")
        trace = []
        print("Working on:", name)
        if os.path.exists(file):
            with open(file, "r") as fd:
                reader = csv.reader(fd)
                trace = [tuple(row) for row in reader]
                trace.pop(0)
                print("\tLoaded", len(trace), "/", len(full_dataset))
        try:
            enumerative_search(pcfgs, trace, method)
        except KeyboardInterrupt:
            should_exit = True
        with open(file, "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(
                ["Solved", "Time (in s)", "Programs Generated", "Solution found"]
            )
            writer.writerows(trace)

        if should_exit:
            break

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({"font.size": 14})
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    plt.style.use("seaborn-colorblind")

    import numpy as np

    ax1 = plt.subplot(1, 2, 1)
    plt.xlabel("Time (in s)")
    plt.ylabel("Tasks Completed")

    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel("# Programs")
    plt.ylabel("Tasks Completed")

    max_time, max_programs = 0, 0
    for name, method in methods:
        file = os.path.join(output_folder, f"{dataset}_{name}.csv")
        trace = []

        if os.path.exists(file):
            with open(file, "r") as fd:
                reader = csv.reader(fd)
                trace = [tuple(row) for row in reader]
                trace.pop(0)
                trace = [
                    (row[0] == "True", float(row[1]), int(row[2])) for row in trace
                ]
            # Plot tasks wrt time
            trace_time = sorted(trace, key=lambda x: x[1])
            cum_sol1, cum_time = np.cumsum([row[0] for row in trace_time]), np.cumsum(
                [row[1] for row in trace_time]
            )
            max_time = max(max_time, cum_time[-1])
            ax1.plot(cum_time, cum_sol1, label=name.capitalize())
            # Plot tasks wrt programs
            trace_programs = sorted(trace, key=lambda x: x[2])
            cum_sol2, cum_programs = np.cumsum(
                [row[0] for row in trace_programs]
            ), np.cumsum([row[2] for row in trace_programs])
            max_programs = max(max_programs, cum_programs[-1])
            ax2.plot(cum_programs, cum_sol2, label=name.capitalize())
            print(name, "solved", cum_sol2[-1], "/", len(trace))
    ax1.hlines(
        [len(full_dataset)],
        xmin=0,
        xmax=max_time,
        label="All tasks",
        color="k",
        linestyles="dashed",
    )
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, len(full_dataset) + 10)
    ax2.hlines(
        [len(full_dataset)],
        xmin=0,
        xmax=max_programs,
        label="All tasks",
        color="k",
        linestyles="dashed",
    )
    ax2.set_xlim(0, max_programs)
    ax2.set_ylim(0, len(full_dataset) + 10)
    ax1.legend()
    ax2.legend()
    plt.show()
