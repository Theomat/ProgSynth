from typing import Dict, List, Optional, Tuple, Union
from synth import Dataset
from synth.specification import (
    PBE,
    CompoundSpecification,
    Example,
    SketchedSpecification,
    TaskSpecification,
)
import json

from synth.task import Task

dataset_file = "./examples/pbe/transduction/dataset/constants.json"

tasks = []
with open(dataset_file) as fd:
    lst = json.load(fd)
    for el in lst:
        spec = PBE([Example(ex["inputs"], ex["output"]) for ex in el["examples"]])
        task = Task[PBE](spec.guess_type(), specification=spec, metadata=el["metadata"])
        tasks.append(task)

dataset: Dataset[TaskSpecification] = Dataset(tasks)


memory: Dict[Tuple[int, ...], List[Union[str, None]]] = {}


def memoised_find_constants(strings: List[str], indices: List[str]):
    key = tuple(indices)
    if key in memory:
        return memory[key]
    else:
        out = find_constants(strings, indices)
        memory[key] = out
        return out


def find_constants(
    strings: List[str], my_indices: Optional[List[int]] = None
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
        possibles = [
            memoised_find_constants(
                strings, [int(i == z) + j for z, j in enumerate(indices)]
            )
            for i in iterator
        ]
    best = [(sum(len(s) for s in l if l is not None), l) for l in possibles]
    best.sort(reverse=True)
    found = best[0][1]
    # if has_found_constant:
    #     print(f"\tfound:\"{constant}\" from", my_indices)
    return [constant] + found if has_found_constant else found


for i, task in enumerate(dataset):
    memory.clear()
    pbe: PBE = task.specification.get_specification(PBE)
    assert pbe is not None
    print("=" * 60)
    print(f"[N°{i}] {task.metadata['name']}")
    print(
        "Constants:",
        find_constants([pbe.examples[i].output for i in range(len(pbe.examples))]),
    )
    print("Sample:", pbe.examples[0].output)
    print()
