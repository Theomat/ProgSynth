from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from examples.pbe.transduction.knowledge_graph.kg_path_finder import (
    build_wrapper,
    choose_best_path,
    find_paths_from_level,
)
from synth import Dataset
from synth.specification import (
    PBE,
)
import argparse

MAX_EXAMPLES_ANALYSED = 5


def memoised_find_constants(
    strings: List[str],
    indices: List[str],
    memory: Dict[Tuple[int, ...], List[Union[str, None]]],
):
    key = tuple(indices)
    if key in memory:
        return memory[key]
    else:
        out = find_constants(strings, indices, memory)
        memory[key] = out
        return out


def find_constants(
    strings: List[str],
    my_indices: Optional[List[int]] = None,
    memory: Optional[Dict[Tuple[int, ...], List[Union[str, None]]]] = None,
) -> List[Union[str, None]]:

    indices = my_indices or [0 for _ in strings]
    start = indices[0]
    iterator = list(range(len(strings)))
    if any(indices[i] >= len(strings[i]) for i in iterator):
        return []
    all_agree = 1 == len({strings[i][indices[i]] for i in iterator})
    last_call = False
    while all_agree:
        indices = [x + 1 for x in indices]
        if any(indices[i] >= len(strings[i]) for i in iterator):
            last_call = True
            break
        all_agree = 1 == len({strings[i][indices[i]] for i in iterator})
        # if not all_agree:
        #     print(f"\t\t does not agree after:\"{strings[0][start:indices[0]]}\" with:", {
        #         strings[i][indices[i]] for i in iterator})
    constant = strings[0][start : indices[0]]
    has_found_constant = len(constant) > 0
    if last_call:
        return [constant] if has_found_constant else [None]
    else:
        real_memory = memory or {}
        possibles = [
            memoised_find_constants(
                strings, [int(i == z) + j for z, j in enumerate(indices)], real_memory
            )
            for i in iterator
        ]
    best = [
        (sum(len(s) for s in filter_constants(l) if l is not None), l)
        for l in possibles
    ]
    best.sort(reverse=True)
    found = best[0][1]
    # if has_found_constant:
    #     print(f"\tfound:\"{constant}\" from", my_indices)
    return [constant] + found if has_found_constant else found


def sketch(output: str, constants: List[str]) -> List[str]:
    out = []
    i, j, last = 0, 0, 0
    while i < len(output) and j < len(constants):
        if output[i:].startswith(constants[j]):
            if i - last > 0:
                out.append(output[last:i])
            else:
                out.append("")
            i += len(constants[j])
            last = i
            j += 1
            if i >= len(output):
                out.append("")
            if j == len(constants):
                break
        else:
            i += 1
    if j == len(constants):
        if i < len(output):
            out.append(output[i:])
    return out


ALPHA_NUMERIC = "0123456789azertyuiopqsdfghjklmwxcvbn"


def filter_constants(constants: List[str]) -> List[str]:
    return [x for x in constants if not (len(x) == 1 and x.lower() in ALPHA_NUMERIC)]


to_update = set([])
if __name__ == "__main__":
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Preporocess transduction tasks to find constants."
    )

    argument_default_values = {
        "file": "constants.pickle",
        "endpoint": "http://192.168.1.20:9999/blazegraph/namespace/kb/sparql",
    }

    argument_parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=argument_default_values["file"],
        help="Source dataset file (default: " + argument_default_values["file"] + ")",
    )
    argument_parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        default=argument_default_values["endpoint"],
        help="SPARQL endpoint (default: " + argument_default_values["endpoint"] + ")",
    )
    args = argument_parser.parse_args()
    dataset_file = args.file
    dataset = Dataset.load(dataset_file)
    wrapper = build_wrapper(args.endpoint)
    print("Loaded tasks!")
    solvable = 0
    for i, task in enumerate(dataset):
        if task.metadata["constant_post_processing"] != 0:
            continue
        pbe: PBE = task.specification.get_specification(PBE)
        assert pbe is not None
        solvable += 1
        print("=" * 60)
        print(f"[N°{i}] {task.metadata['name']}")
        print("Sample:", pbe.examples[0].output)
        constants = task.metadata.get("constants", None)
        if constants is None or task.metadata["name"] in to_update:
            all_possibles = [
                filter_constants(
                    find_constants(
                        [
                            pbe.examples[i + k].output
                            for i in range(
                                min(MAX_EXAMPLES_ANALYSED, len(pbe.examples))
                            )
                        ]
                    )
                )
                for k in range(max(1, len(pbe.examples) - MAX_EXAMPLES_ANALYSED))
            ]
            t = sorted([(len(l), l) for l in all_possibles])
            constants = t[0][1]
            task.metadata["constants"] = constants
            dataset.save(dataset_file)
        print("Constants:", constants)

        new_pseudo_tasks = defaultdict(list)
        for i in range(len(pbe.examples)):
            subtasks = sketch(pbe.examples[i].output, constants)
            if i == 0:
                print("\tsample sketch:", subtasks)
            for j in range(len(subtasks)):
                new_pseudo_tasks[j].append((pbe.examples[i].inputs[0], subtasks[j]))
        for query_task, pairs in new_pseudo_tasks.items():
            print("\tPairs:", pairs)
            d = task.metadata["knowledge_graph_relationship"] - 1
            paths = find_paths_from_level(pairs, wrapper, d)
            if paths:
                if len(paths) > 1:
                    paths = [choose_best_path(paths, pairs, wrapper)]
                for path in paths:
                    print("\t\tstart->" + "->".join(path) + "->end")
            else:
                print("\tFound no path for relationship level", d)
        print()

        constants_in = task.metadata.get("constants_in", None)
        if constants_in is None or task.metadata["name"] in to_update:
            all_possibles = [
                filter_constants(
                    find_constants(
                        [
                            pbe.examples[i + k].inputs[0]
                            for i in range(
                                min(MAX_EXAMPLES_ANALYSED, len(pbe.examples))
                            )
                        ]
                    )
                )
                for k in range(max(1, len(pbe.examples) - MAX_EXAMPLES_ANALYSED))
            ]
            t = sorted([(len(l), l) for l in all_possibles])
            constants_in = t[0][1]
            task.metadata["constants_in"] = constants_in
            dataset.save(dataset_file)
        print("Constants Input:", constants_in)
    print("Found", solvable, "solvable tasks")
