import atexit
from collections import defaultdict
import os
import sys
from typing import Callable, Iterable, List, Optional, Tuple, Union
import csv

import tqdm

import torch

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from model_loader import (
    add_model_choice_arg,
    instantiate_predictor,
)


from synth import Dataset, PBE, Task
from synth.nn import free_pytorch_memory
from synth.filter import add_dfta_constraints
from synth.semantic import DSLEvaluator
from synth.semantic.evaluator import DSLEvaluatorWithConstant
from synth.specification import Example, PBEWithConstants
from synth.syntax import (
    CFG,
    UCFG,
    ProbDetGrammar,
    ProbUGrammar,
    enumerate_prob_grammar,
    enumerate_prob_u_grammar,
    enumerate_bucket_prob_grammar,
    enumerate_bucket_prob_u_grammar,
    DSL,
    Program,
)
from synth.syntax.grammars.enumeration.heap_search import HSEnumerator
from synth.syntax.program import Function, Primitive, Variable
from synth.syntax.type_system import STRING, Arrow
from synth.utils import chrono, load_object, save_object


import argparse

parser = argparse.ArgumentParser(description="Evaluate model prediction")
parser.add_argument("-m", "--model", default="", type=str, help="model file")
add_dataset_choice_arg(parser)
parser.add_argument(
    "-s",
    "--search",
    type=str,
    default="heap_search",
    help="enumeration algorithm (default: heap_search)",
)
parser.add_argument(
    "--method",
    type=str,
    default="base",
    help="used method (default: base)",
)
parser.add_argument(
    "--predict",
    action="store_true",
    help="only do the PCFG prediction part",
)
add_dsl_choice_arg(parser)
add_model_choice_arg(parser)
parser.add_argument(
    "-o", "--output", type=str, default="./", help="output folder (default: './')"
)
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
parser.add_argument(
    "-t", "--timeout", type=float, default=300, help="task timeout in s (default: 300)"
)


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
search_algo: str = parameters.search
method: str = parameters.method
output_folder: str = parameters.output
model_file: str = parameters.model
task_timeout: float = parameters.timeout
batch_size: int = parameters.batch_size
constrained: bool = parameters.constrained
predict_only: bool = parameters.predict
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
if search_algo == "heap_search":
    custom_enumerate = (
        enumerate_prob_grammar if not constrained else enumerate_prob_u_grammar
    )
elif search_algo == "bucket_search":
    custom_enumerate = (
        lambda x: enumerate_bucket_prob_grammar(x, 3)
        if not constrained
        else enumerate_bucket_prob_u_grammar(x, 3)
    )
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

supported_type_requests = Dataset.load(support).type_requests() if support else None

# ================================
# Load constants specific to dataset
# ================================


def load_dsl_and_dataset() -> Tuple[
    Dataset[PBE], DSL, DSLEvaluatorWithConstant, List[int], str, List[str]
]:
    dsl_module = load_DSL(dsl_name)
    dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon
    constraints = []
    if constrained and hasattr(dsl_module, "constraints"):
        constraints = dsl_module.constraints
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
    return full_dataset, dsl, evaluator, lexicon, model_name, constraints


# Produce PCFGS ==========================================================
@torch.no_grad()
def produce_pcfgs(
    full_dataset: Dataset[PBE], dsl: DSL, lexicon: List[int], constraints: List[str]
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
    if predict_only:
        return pcfgs
    del predictor
    free_pytorch_memory()
    return pcfgs


def save(trace: Iterable) -> None:
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


# Enumeration methods =====================================================
def enumerative_search(
    dataset: Dataset[PBE],
    evaluator: DSLEvaluatorWithConstant,
    pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]],
    trace: List[Tuple[bool, float]],
    method: Callable[
        [DSLEvaluatorWithConstant, Task[PBE], Union[ProbDetGrammar, ProbUGrammar]],
        Tuple[bool, float, int, Optional[Program]],
    ],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
) -> None:

    start = len(trace)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks", smoothing=0)
    i = 0
    solved = 0
    total = 0
    tasks = [
        t
        for t in dataset.tasks
        if supported_type_requests is None or t.type_request in supported_type_requests
    ]
    for task, pcfg in zip(tasks[start:], pcfgs[start:]):
        total += 1
        try:
            out = method(evaluator, task, pcfg, custom_enumerate)
            trace.append(out)
            if out[0]:
                solved += 1
        except KeyboardInterrupt:
            break
        pbar.update(1)
        evaluator.clear_cache()
        # print("Cache hit:", evaluator.cache_hit_rate)
        # print("Programs tried:", trace[len(trace) - 1][2])
        if i % 10 == 0:
            pbar.set_postfix_str("Saving...")
            save(trace)
        pbar.set_postfix_str(f"Solved {solved}/{total}")

    pbar.close()


def base(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
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


def semantic_base(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program], Optional[float]]:
    time = 0.0
    programs = 0
    with chrono.clock("search.semantic_base") as c:
        enumerator = custom_enumerate(pcfg)
        for program in enumerator:
            time = c.elapsed_time()
            if time >= task_timeout:
                return (False, time, programs, None, None)
            programs += 1
            failed = False
            for ex in task.specification.examples:
                out = evaluator.eval(program, ex.inputs)
                failed = failed or out != ex.output
            if not failed:
                return (
                    True,
                    c.elapsed_time(),
                    programs,
                    program,
                    pcfg.probability(program),
                )
    return (False, time, programs, None, None)


def semantic_equivalence(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program], Optional[float]]:
    time = 0.0
    programs = 0
    with chrono.clock("search.semantic_equivalence") as c:
        results = {}
        enumerator = custom_enumerate(pcfg)
        merged = 0
        for program in enumerator:
            time = c.elapsed_time()
            if time >= task_timeout:
                return (False, time, programs, None, None)
            programs += 1
            failed = False
            outputs = None
            for ex in task.specification.examples:
                out = evaluator.eval(program, ex.inputs)
                failed |= out != ex.output
                if isinstance(out, list):
                    outputs = (outputs, tuple(out))
                else:
                    outputs = (outputs, out)
            if not failed:
                return (
                    True,
                    c.elapsed_time(),
                    programs,
                    program,
                    merged,
                )
            else:
                original = results.get(outputs)
                if original is not None:
                    enumerator.merge_program(original, program)
                    merged += 1
                else:
                    results[outputs] = program
    return (False, time, programs, None, merged)


def constants_injector(
    evaluator: DSLEvaluatorWithConstant,
    task: Task[PBEWithConstants],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program]]:
    time = 0.0
    programs = 0
    constants_in = task.specification.constants_in
    if len(constants_in) == 0:
        constants_in.append("")
    constants_out = task.specification.constants_out
    if len(constants_out) == 0:
        constants_out.append("")
    # program = task.solution
    # if program == None:
    #     return (False, time, programs, None, None)
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
                return (
                    True,
                    c.elapsed_time(),
                    programs,
                    program,
                    pcfg.probability(program),
                )
    return (False, time, programs, None, None)


def sketched_base(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    custom_enumerate: Callable[[Union[ProbDetGrammar, ProbUGrammar]], HSEnumerator],
) -> Tuple[bool, float, int, Optional[Program]]:
    from examples.pbe.transduction.knowledge_graph.kg_path_finder import (
        build_wrapper,
        choose_best_path,
        find_paths_from_level,
    )
    from examples.pbe.transduction.knowledge_graph.preprocess_tasks import sketch

    programs = 0
    global task_timeout
    if task.metadata.get("constants", None) is not None:
        original_timeout = task_timeout
        verbose = False
        # (
        #     task.metadata["constant_post_processing"] == 0
        #     and task.metadata["constant_detection"] == 0
        #     and task.metadata["knowledge_graph_relationship"] > 0
        # )
        if verbose:
            print("should solve:", task.metadata.get("name", "???"))
        with chrono.clock("additional") as c:
            wrapper = build_wrapper(
                "http://192.168.1.20:9999/blazegraph/namespace/kb/sparql"
            )
            constants = task.metadata.get("constants", None)
            constants_in = task.metadata.get("constants_in", [])
            pbe = task.specification
            new_pseudo_tasks = defaultdict(lambda: defaultdict(list))
            # print("working on:", task.metadata["name"])
            # print("constants out.:", constants)
            # print("constants inp.:", constants_in)
            true_inputs = (
                [
                    sketch(pbe.examples[i].inputs[0], constants_in)
                    for i in range(len(pbe.examples))
                ]
                if constants_in
                else [pbe.examples[i].inputs for i in range(len(pbe.examples))]
            )
            # print("true_inputs:", true_inputs)
            n = len(true_inputs[0])
            for i in range(len(pbe.examples)):
                subtasks = sketch(pbe.examples[i].output, constants)
                for j in range(len(subtasks)):
                    for k in range(n):
                        new_pseudo_tasks[j][k].append((true_inputs[i][k], subtasks[j]))
            solution_part = []
            prob = 1
            for j, possibles in new_pseudo_tasks.items():
                any_solved = False
                relevant_alternatives = {
                    k: pairs
                    for k, pairs in possibles.items()
                    if not all(len(out) == 0 for _, out in pairs)
                    and not all(len(inp) == 0 for inp, _ in pairs)
                }
                subn = len(relevant_alternatives)
                if subn == 0:
                    continue
                # print(
                #     f"\t\tpart[{j}] before:{possibles}")
                # print(
                #     f"\t\tpart[{j}] before:{len(possibles)} after:{len(relevant_alternatives)}")
                for k, pairs in relevant_alternatives.items():
                    # print("\tsub task:", pairs)
                    d = task.metadata["knowledge_graph_relationship"] - 1
                    paths = find_paths_from_level(pairs, wrapper, d)
                    # print("\t\tfound paths:", paths)
                    if paths:
                        any_solved = True
                        if len(paths) > 1:
                            paths = [choose_best_path(paths, pairs, wrapper)]
                        custom_input = Variable(0, STRING)
                        if not (k == 0 and k + 1 >= len(constants_in)):
                            custom_input = Function(
                                Primitive(
                                    f"between {constants_in[k] if k > 0 else 'start'} and {constants_in[k + 1] if k + 1 < len(constants_in) else 'end'}",
                                    Arrow(STRING, STRING),
                                ),
                                [custom_input],
                            )
                        solution_part.append(
                            Function(
                                Primitive(
                                    "start->" + "->".join(paths[0]) + "->end",
                                    Arrow(STRING, STRING),
                                ),
                                [custom_input],
                            )
                        )
                        if verbose:
                            print(
                                "\tresult:", "start->" + "->".join(paths[0]) + "->end"
                            )
                    else:
                        sub_task = Task(
                            task.type_request,
                            PBE(
                                [
                                    Example([pairs[i][0]], pairs[i][1])
                                    for i in range(len(pbe.examples))
                                ],
                            ),
                        )
                        task_timeout = original_timeout - c.elapsed_time()
                        task_timeout /= subn
                        if verbose:
                            print(
                                "\tsolving with timeout",
                                task_timeout,
                                "s :",
                                sub_task.specification.examples,
                            )

                        (
                            solved,
                            _,
                            enumerated,
                            partial_sol,
                            part_prob,
                        ) = base(evaluator, sub_task, pcfg, custom_enumerate)
                        task_timeout = original_timeout
                        if verbose:
                            print("\tresult:", solved, partial_sol)
                        if c.elapsed_time() >= task_timeout:
                            return (False, c.elapsed_time(), programs, None, None)
                        if solved:
                            any_solved = True
                            prob *= part_prob
                            solution_part.append(partial_sol)
                        programs += enumerated
                    if any_solved:
                        break
                if not any_solved:
                    return False, c.elapsed_time(), programs, None, None
            # Convert back to a program
            some_output: str = pbe.examples[0].output
            start_cste = len(constants) > 0 and some_output.startswith(constants[0])
            i = 0
            concat_type = STRING
            if start_cste:
                arguments = [Primitive('"' + constants[0] + '"', STRING)]
                for cste in constants[1:]:
                    arguments.append(solution_part[i])
                    concat_type = Arrow(concat_type, STRING)
                    arguments.append(Primitive('"' + cste + '"', STRING))
                    concat_type = Arrow(concat_type, STRING)
                    i += 1
                if i < len(solution_part):
                    arguments.append(solution_part[i])
                    concat_type = Arrow(concat_type, STRING)

            else:
                arguments = [solution_part.pop(0)]
                for cste in constants:
                    arguments.append(Primitive('"' + cste + '"', STRING))
                    concat_type = Arrow(concat_type, STRING)
                    if i < len(solution_part):
                        arguments.append(solution_part[i])
                        concat_type = Arrow(concat_type, STRING)
                    i += 1
                if i < len(solution_part):
                    arguments.append(solution_part[i])
                    concat_type = Arrow(concat_type, STRING)
            end_solution = (
                Function(Primitive("concat", concat_type), arguments)
                if len(arguments) > 1
                else arguments[0]
            )
            return True, c.elapsed_time(), programs, end_solution, prob

    else:
        # print("timeout:", task_timeout)
        if task.specification.get_specification(PBEWithConstants) is not None:
            return constants_injector(evaluator, task, pcfg, custom_enumerate)
        else:
            return base(evaluator, task, pcfg, custom_enumerate)


# Main ====================================================================

if __name__ == "__main__":
    (
        full_dataset,
        dsl,
        evaluator,
        lexicon,
        model_name,
        constraints,
    ) = load_dsl_and_dataset()

    METHODS = {
        "base": base,
        "sem.equiv": semantic_equivalence,
        "sem.base": semantic_base,
        "wikicoder": sketched_base,
        "constant": constants_injector,
    }
    method_fn = METHODS[method]

    pcfgs = produce_pcfgs(full_dataset, dsl, lexicon, constraints)
    if not predict_only:
        file = os.path.join(
            output_folder, f"{dataset_name}_{model_name}_{search_algo}_{method}.csv"
        )
        trace = []
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
        enumerative_search(
            full_dataset, evaluator, pcfgs, trace, method_fn, custom_enumerate
        )
        save(trace)
        print("csv file was saved as:", file)
