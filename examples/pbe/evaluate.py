import atexit
import os
from glob import glob
import sys
from typing import Callable, List, Optional, Tuple
import csv
import pickle

import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from synth import Dataset, PBE, Task
from synth.nn import BigramsPredictorLayer, Task2Tensor, free_pytorch_memory
from synth.pbe import IOEncoder
from synth.semantic import DSLEvaluator
from synth.semantic.evaluator import DSLEvaluatorWithConstant
from synth.specification import PBEWithConstants
from synth.syntax import (
    ConcreteCFG,
    ConcretePCFG,
    enumerate_pcfg,
    enumerate_bucket_pcfg,
    DSL,
    Program,
)
from synth.syntax.grammars.heap_search import HSEnumerator
from synth.utils import chrono


DREAMCODER = "dreamcoder"
DEEPCODER = "deepcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"

import argparse

parser = argparse.ArgumentParser(description="Evaluate model prediction")
parser.add_argument("-m", "--model", default="", type=str, help="model file")
parser.add_argument(
    "-p", "--plot", default=False, action="store_true", help="only plot results"
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default=DEEPCODER + ".pickle",
    help="dataset (default: deepcoder.pickle)",
)
parser.add_argument(
    "-s",
    "--search",
    type=str,
    default="heap_search",
    help="enumeration algorithm (default: heap_search)",
)
parser.add_argument(
    "--dsl",
    type=str,
    default=DEEPCODER,
    choices=[DEEPCODER, DREAMCODER, REGEXP, CALCULATOR, TRANSDUCTION],
    help="dsl (default: deepcoder)",
)
parser.add_argument(
    "-o", "--output", type=str, default="./", help="output folder (default: './')"
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
g = parser.add_argument_group("pcfg prediction parameter")
g.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=16,
    help="batch size to compute PCFGs (default: 16)",
)
parser.add_argument(
    "-t", "--timeout", type=float, default=300, help="task timeout in s (default: 300)"
)


parameters = parser.parse_args()
dataset_file: str = parameters.dataset
dsl_name: str = parameters.dsl
search_algo: str = parameters.search
output_folder: str = parameters.output
model_file: str = parameters.model
variable_probability: float = parameters.var_prob
encoding_dimension: int = parameters.encoding_dimension
hidden_size: int = parameters.hidden_size
task_timeout: float = parameters.timeout
batch_size: int = parameters.batch_size
plot_only: bool = parameters.plot


if not plot_only and (not os.path.exists(model_file) or not os.path.isfile(model_file)):
    print("Model must be a valid model file!", file=sys.stderr)
    sys.exit(1)
elif not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)

if search_algo == "heap_search":
    custom_enumerate = enumerate_pcfg
elif search_algo == "bucket_search":
    custom_enumerate = lambda x: enumerate_bucket_pcfg(x, 3)
    # TODO: add parameter for bucket_search size
else:
    print(
        "search algorithm must be a valid name (heap_search / bucket_search)!",
        file=sys.stderr,
    )
    sys.exit(1)

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

# ================================
# Load constants specific to dataset
# ================================


def load_dataset() -> Tuple[
    Dataset[PBE], DSL, DSLEvaluatorWithConstant, List[int], str
]:
    if dsl_name == DEEPCODER:
        from deepcoder.deepcoder import dsl, evaluator, lexicon
    elif dsl_name == DREAMCODER:
        from dreamcoder.dreamcoder import dsl, evaluator, lexicon
    elif dsl_name == REGEXP:
        from regexp.regexp import dsl, evaluator, lexicon
    elif dsl_name == CALCULATOR:
        from calculator.calculator import dsl, evaluator, lexicon
    elif dsl_name == TRANSDUCTION:
        from transduction.transduction import dsl, evaluator, lexicon
    else:
        print("Unknown dsl:", dsl_name, file=sys.stderr)
        sys.exit(0)
    # ================================
    # Load dataset
    # ================================
    # Load dataset
    print(f"Loading {dataset_file}...", end="")
    with chrono.clock("dataset.load") as c:
        full_dataset = Dataset.load(dataset_file)
        print("done in", c.elapsed_time(), "s")

    if plot_only:
        model_name = ""
    else:
        start_index = (
            0
            if not os.path.sep in model_file
            else (len(model_file) - model_file[::-1].index(os.path.sep))
        )
        model_name = model_file[start_index : model_file.index(".", start_index)]
    return full_dataset, dsl, evaluator, lexicon, model_name


# Produce PCFGS ==========================================================
@torch.no_grad()
def produce_pcfgs(
    full_dataset: Dataset[PBE], dsl: DSL, lexicon: List[int]
) -> List[ConcreteCFG]:
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
    if all(task.solution is not None for task in full_dataset):
        max_depth = max(task.solution.depth() for task in full_dataset)
    else:
        max_depth = 10  # TODO: set as parameter
    cfgs = [ConcreteCFG.from_dsl(dsl, t, max_depth) for t in all_type_requests]

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

    predictor = MyPredictor(hidden_size)
    predictor.load_state_dict(torch.load(model_file))
    predictor = predictor.to(device)
    predictor.eval()
    # ================================
    # Predict PCFG
    # ================================
    def save_pcfgs() -> None:
        with open(file, "wb") as fd:
            pickle.dump(pcfgs, fd)

    atexit.register(save_pcfgs)

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
    atexit.unregister(save_pcfgs)
    del predictor
    free_pytorch_memory()
    return pcfgs


# Enumeration methods =====================================================
def enumerative_search(
    dataset: Dataset[PBE],
    evaluator: DSLEvaluatorWithConstant,
    pcfgs: List[ConcretePCFG],
    trace: List[Tuple[bool, float]],
    method: Callable[
        [DSLEvaluatorWithConstant, Task[PBE], ConcretePCFG],
        Tuple[bool, float, int, Optional[Program]],
    ],
    custom_enumerate: Callable[[ConcretePCFG], HSEnumerator],
) -> None:

    start = len(trace)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks", smoothing=0)
    for task, pcfg in zip(dataset.tasks[start:], pcfgs[start:]):
        trace.append(method(evaluator, task, pcfg, custom_enumerate))
        pbar.update(1)
        # print("Cache hit:", evaluator.cache_hit_rate)
        # print("Programs tried:", trace[len(trace) - 1][2])
    pbar.close()


def base(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: ConcretePCFG,
    custom_enumerate: Callable[[ConcretePCFG], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program]]:
    time = 0.0
    programs = 0
    with chrono.clock("search.base") as c:

        for program in custom_enumerate(pcfg):
            time = c.elapsed_time()
            if time >= task_timeout:
                return (False, time, programs, None, None)
            programs += 1
            failed = False
            for ex in task.specification.examples:
                if evaluator.eval(program, ex.inputs) != ex.output:
                    failed = True
                    break
            if not failed:
                return (
                    True,
                    c.elapsed_time(),
                    programs,
                    program,
                    pcfg.probability(program),
                )
    return (False, time, programs, None, None)


def constants_injector(
    evaluator: DSLEvaluatorWithConstant,
    task: Task[PBEWithConstants],
    pcfg: ConcretePCFG,
    custom_enumerate: Callable[[ConcretePCFG], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program]]:
    time = 0.0
    programs = 0
    constants_in = task.specification.constants_in
    if len(constants_in) == 0:
        constants_in.append("")
    constants_out = task.specification.constants_out
    if len(constants_out) == 0:
        constants_out.append("")
    name = task.metadata["name"]
    program = task.solution
    if program == None:
        return (False, time, programs, None, None)
    with chrono.clock("search.constant_injector") as c:

        # print("\n-----------------------")
        # print(name)
        for program in custom_enumerate(pcfg):
            time = c.elapsed_time()
            if time >= task_timeout:
                # print("TIMEOUT\n\n")
                return (False, time, programs, None, None)
            programs += 1
            found = False
            counter = 0
            for ex in task.specification.examples:
                found = False
                for cons_in in constants_in:
                    for cons_out in constants_out:
                        if (
                            evaluator.eval_with_constant(
                                program, ex.inputs, cons_in, cons_out
                            )
                            == ex.output
                        ):
                            found = True
                            counter += 1
                            break
                    if found:
                        break
                if not found:
                    break
            if found:
                # print("Solution found.\n")
                # print("\t", program)
                # print(
                #    "\nWorking for all ",
                #    counter,
                #    "/",
                #    len(task.specification.examples),
                #    " examples in ",
                #    time,
                #    "/",
                #    task_timeout,
                #    "s.",
                # )
                return (
                    True,
                    c.elapsed_time(),
                    programs,
                    program,
                    pcfg.probability(program),
                )
    return (False, time, programs, None, None)


# Main ====================================================================

if __name__ == "__main__":
    methods = [
        ("constants_injector", constants_injector),
    ]

    full_dataset, dsl, evaluator, lexicon, model_name = load_dataset()

    if not plot_only:
        pcfgs = produce_pcfgs(full_dataset, dsl, lexicon)
        should_exit = False
        for name, method in methods:
            file = os.path.join(
                output_folder, f"{dataset_name}_{model_name}_{search_algo}_{name}.csv"
            )
            trace = []
            print("Working on:", name)
            if os.path.exists(file):
                with open(file, "r") as fd:
                    reader = csv.reader(fd)
                    trace = [tuple(row) for row in reader]
                    trace.pop(0)
                    print(
                        "\tLoaded",
                        len(trace),
                        "/",
                        len(full_dataset),
                        "(",
                        int(len(trace) * 100 / len(full_dataset)),
                        "%)",
                    )
            try:
                enumerative_search(
                    full_dataset, evaluator, pcfgs, trace, method, custom_enumerate
                )
            except Exception as e:
                print(e)
                should_exit = True
            with open(file, "w") as fd:
                writer = csv.writer(fd)
                writer.writerow(
                    [
                        "Solved",
                        "Time (in s)",
                        "Programs Generated",
                        "Solution found",
                        "Program probability",
                    ]
                )
                writer.writerows(trace)
            print("csv file is saved.")
            if should_exit:
                break

    import numpy as np
    import matplotlib.pyplot as plt

    import pltpublish as pub

    pub.setup()

    ax1 = plt.subplot(1, 2, 1)
    plt.xlabel("Time (in s)")
    plt.ylabel("Tasks Completed")
    plt.grid()

    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel("# Programs")
    plt.ylabel("Tasks Completed")
    plt.grid()

    max_time, max_programs = 0, 0
    for file in glob(os.path.join(output_folder, "*.csv")):
        file = os.path.relpath(file, output_folder)
        if not file.startswith(dataset_name):
            continue
        name = file[len(dataset_name) : -4]
        if "_" not in name:
            continue
        name = name[name.index("_") + 1 :].replace("_", " ")
        trace = []
        with open(file, "r") as fd:
            reader = csv.reader(fd)
            trace = [tuple(row) for row in reader]
            trace.pop(0)
            trace = [(row[0] == "True", float(row[1]), int(row[2])) for row in trace]
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
