import json
import os
import pickle
import atexit
import sys

from typing import Tuple, Any, Dict, List
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import tqdm
from synth.semantic.evaluator import DSLEvaluatorWithConstant
from synth.specification import PBE
import math
from synth.syntax import DSL
from synth.nn.pcfg_predictor import ConcreteCFG
from synth.syntax.grammars.concrete_pcfg import ConcretePCFG
from synth.syntax.program import Function, Primitive, Program
from synth.task import Dataset
from synth.utils import chrono
from synth.syntax import enumerate_pcfg
from synth import Dataset, PBE, Task
from synth.nn import BigramsPredictorLayer, Task2Tensor, free_pytorch_memory
from synth.pbe import IOEncoder

MAX_TESTS = 400000


def __log_e__(n: int):
    return math.log(n)


def __count_programs__(solution: Program, pcfg: ConcretePCFG) -> int:
    if solution is None:
        return {None: None}
    counters: Dict[Program, Tuple[int, int]] = {}
    for sub_program in solution.depth_first_iter():
        found: bool = False
        if isinstance(sub_program, Function):
            counter = 0
            for program in enumerate_pcfg(pcfg):
                counter += 1
                if program == sub_program or counter >= MAX_TESTS:
                    counters[str(sub_program)] = (counter, sub_program.depth())
                    found = True
                    break
            if (
                not found
            ):  # can happen when function signature does not correspond to pcfg (thus, cannot be found)
                counters[str(sub_program)] = (MAX_TESTS, sub_program.depth())
    return counters


def __analyse_primitives__(
    dsl: DSL,
    dataset: Dataset[PBE],
    pcfg: ConcretePCFG,
    state: str,
    jsonobject: List[Dict[Primitive, Any]],
) -> None:
    print("Analysing data from state '", state, "'....", end="")
    with chrono.clock("analyse." + state) as c:
        task_number = 1
        pbar = tqdm.tqdm(
            total=len(pcfg),
            desc="statistical analysis of primitive in state '" + state + "'",
        )
        for task, pcfg in zip(dataset.tasks, pcfg):
            solution = task.solution
            counter = __count_programs__(solution, pcfg)
            print("----------------")
            print("Solutions found: ", counter)
            obj = {"task": task_number, "programs": [counter]}
            task_number += 1
            jsonobject.append(obj)
            pbar.update(1)
        pbar.close()


def load_dataset(dataset_file: str) -> Dataset[PBE]:
    print(f"Loading {dataset_file}...", end="")
    with chrono.clock("dataset.load") as c:
        full_dataset = Dataset.load(dataset_file)
        print("done in", c.elapsed_time(), "s")
    return full_dataset


def analyse_dsl(
    dsl: DSL, dataset: str, lexicon: List[int], model: str, state: str
) -> Dict[Any, Any]:
    stats = []
    full_dataset = load_dataset(dataset)
    pcfgs = produce_pcfgs(full_dataset, dataset, dsl, lexicon, model)
    __analyse_primitives__(dsl, full_dataset, pcfgs, state, stats)
    return stats


def load_dsl(dsl_name: str) -> Tuple[DSL, DSLEvaluatorWithConstant, List[int]]:
    if dsl_name == DEEPCODER:
        from examples.pbe.deepcoder.deepcoder import dsl, evaluator, lexicon
    elif dsl_name == DREAMCODER:
        from examples.pbe.dreamcoder.dreamcoder import dsl, evaluator, lexicon
    elif dsl_name == REGEXP:
        from examples.pbe.regexp.regexp import dsl, evaluator, lexicon
    elif dsl_name == CALCULATOR:
        from examples.pbe.calculator.calculator import dsl, evaluator, lexicon
    elif dsl_name == TRANSDUCTION:
        from examples.pbe.transduction.transduction import dsl, evaluator, lexicon
    else:
        print("Unknown dsl:", dsl_name, file=sys.stderr)
        sys.exit(0)
    return dsl, evaluator, lexicon


def dataset_to_pcfg_bigram(
    full_dataset: Dataset[PBE], cfg: ConcreteCFG, filter: bool = False
) -> ConcretePCFG:
    """
    - filter (bool, default=False) - compute pcfg only on tasks with the same type request as the cfg's
    """
    samples = [
        task.solution
        for task in full_dataset.tasks
        if task.solution and (not filter or cfg.type_request == task.type_request)
    ]
    return ConcretePCFG.from_samples_bigram(cfg, samples)


def produce_bigrams(full_dataset: Dataset[PBE], dsl: DSL) -> List[ConcretePCFG]:
    all_type_requests = list(full_dataset.type_requests())
    if all(task.solution is not None for task in full_dataset):
        max_depth = max(task.solution.depth() for task in full_dataset)
    else:
        max_depth = 10
    cfgs = [ConcreteCFG.from_dsl(dsl, t, max_depth) for t in all_type_requests]
    types_pcfgs = [dataset_to_pcfg_bigram(full_dataset, c) for c in cfgs]
    pcfgs = []
    for task in full_dataset.tasks:
        type = task.type_request
        index = all_type_requests.index(type)
        pcfgs.append(types_pcfgs[index])
    return pcfgs


@torch.no_grad()
def produce_pcfgs(
    full_dataset: Dataset[PBE],
    dataset_file: str,
    dsl: DSL,
    lexicon: List[int],
    model_file: str,
) -> List[ConcreteCFG]:
    if model_file == BIGRAM:
        return produce_bigrams(full_dataset, dsl)
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

    start_index = (
        0
        if not os.path.sep in dataset_file
        else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
    )
    dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]
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
            y0, _ = self.rnn(seq)
            y = y0.data
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


if __name__ == "__main__":
    import argparse

    DREAMCODER = "dreamcoder"
    DEEPCODER = "deepcoder"
    REGEXP = "regexp"
    CALCULATOR = "calculator"
    TRANSDUCTION = "transduction"
    BIGRAM = "bigram"

    parser = argparse.ArgumentParser(description="Evaluate model prediction")

    parser.add_argument(
        "-db",
        "--dataset-before",
        type=str,
        default="flashfill_before.pickle",
        help="dataset before modification of DSL (default: flashfill_before.pickle)",
    )
    parser.add_argument(
        "-da",
        "--dataset-after",
        type=str,
        default="flashfill_after.pickle",
        help="dataset after modification of DSL (default: flashfill_after.pickle)",
    )
    parser.add_argument(
        "-mb",
        "--model-before",
        type=str,
        default=BIGRAM,
        help="pcfg model for the state 'before'. Default: bigram",
    )
    parser.add_argument(
        "-ma",
        "--model-after",
        type=str,
        default=BIGRAM,
        help="pcfg model for the state 'after'. Default: bigram",
    )
    parser.add_argument(
        "-dslb",
        "--dsl-before",
        type=str,
        default=DEEPCODER,
        choices=[DEEPCODER, DREAMCODER, REGEXP, CALCULATOR, TRANSDUCTION],
    )
    parser.add_argument(
        "-dsla",
        "--dsl-after",
        type=str,
        default=DEEPCODER,
        choices=[DEEPCODER, DREAMCODER, REGEXP, CALCULATOR, TRANSDUCTION],
    )

    parser.add_argument(
        "-v",
        "--var-prob",
        type=float,
        default=0.2,
        help="variable probability (default: .2)",
    )
    parser.add_argument(
        "-ed",
        "--encoding-dimension",
        type=int,
        default=512,
        help="encoding dimension (default: 512)",
    )
    parser.add_argument(
        "-hd",
        "--hidden-size",
        type=int,
        default=512,
        help="hidden layer size (default: 512)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="batch size to compute PCFGs (default: 16)",
    )

    parameters = parser.parse_args()
    dbefore: str = parameters.dataset_before
    dafter: str = parameters.dataset_after
    model_before: str = parameters.model_before
    model_after: str = parameters.model_after
    dslb: str = parameters.dsl_before
    dsla: str = parameters.dsl_after
    variable_probability: float = parameters.var_prob
    encoding_dimension: int = parameters.encoding_dimension
    hidden_size: int = parameters.hidden_size
    batch_size: int = parameters.batch_size

    dsl_before, evaluator_before, lexicon_before = load_dsl(dslb)
    dsl_after, evaluator_after, lexicon_after = load_dsl(dsla)

    tasks_before = analyse_dsl(
        dsl_before, dbefore, lexicon_before, model_before, "before"
    )
    start_index = (
        0
        if not os.path.sep in dbefore
        else (len(dbefore) - dbefore[::-1].index(os.path.sep))
    )
    json_before = (
        {
            "max_tests": MAX_TESTS,
            "dsl": dslb,
            "dataset": dbefore,
            "model": model_before,
            "tasks": tasks_before,
        },
    )
    filename = dbefore[start_index : dbefore.index(".", start_index)]
    filename += "_analysis_before.json"
    with open(filename, "w") as fd:
        json.dump(json_before, fd, indent=4)
    print("Written result 'before' in file", filename)
    tasks_after = analyse_dsl(dsl_after, dafter, lexicon_after, model_after, "after")
    start_index = (
        0
        if not os.path.sep in dafter
        else (len(dafter) - dafter[::-1].index(os.path.sep))
    )
    json_after = (
        {
            "max_tests": MAX_TESTS,
            "dsl": dsla,
            "dataset": dafter,
            "model": model_after,
            "tasks": tasks_after,
        },
    )
    filename = dafter[start_index : dafter.index(".", start_index)]
    filename += "_analysis_after.json"
    with open(filename, "w") as fd:
        json.dump(json_after, fd, indent=4)
    print("Written result 'after' in file", filename)
