from copy import deepcopy
import json
import numpy
import matplotlib.pyplot as plt
import pltpublish as pub
import tqdm
from typing import Dict, Any, List, Set

from synth.task import T

THRESHOLD = 30000  ## TODO: import from argparse


def load_json(filename: str) -> Dict[Any, Any]:
    data = {}
    with open(filename) as json_file:
        data = json.load(json_file)
    return data[0]


def compare_jsons(json_before: Dict[Any, Any], json_after: Dict[Any, Any]) -> None:
    def make_plot_data(values):
        if len(values) == 0:
            return [[MAX_TESTS]]
        _, max_depth = values[-1]
        depths: List[int] = [0 for _ in range(max_depth - 1)]
        list_ptr: List[int] = [0 for _ in range(max_depth - 1)]
        lists = [[0]]
        for val, depth in values:
            index = depth - 2
            depths[index] += 1
            if len(lists) < depths[index]:
                lists.append(deepcopy(lists[0]))
                lists[-1][-1] = val
            else:
                for l in lists[list_ptr[index] :]:
                    l.append(val)
            list_ptr[index] += 1
        return lists

    pub.setup()
    avg_solution_before = 0
    avg_solution_after = 0
    task_number = 1
    number_printed = 0
    timeout_before = []
    timeout_after = []
    pbar = tqdm.tqdm(total=len(json_before["tasks"]), desc="Comparison")
    for before, after in zip(json_before["tasks"], json_after["tasks"]):
        y1 = make_plot_data(list(before["programs"][0].values()))
        y2 = make_plot_data(list(after["programs"][0].values()))
        pbar.update(1)
        result_before = y1[0][-1]
        result_after = y2[0][-1]
        if result_before >= MAX_TESTS:
            timeout_before.append(task_number)
        if result_after >= MAX_TESTS:
            timeout_after.append(task_number)
        if result_before != MAX_TESTS and result_after != MAX_TESTS:
            avg_solution_before += result_before
            avg_solution_after += result_after
            if number_printed < number_task_print and (
                result_before > THRESHOLD or result_after > THRESHOLD
            ):
                plt.figure()
                for l in y1:
                    plt.plot(l, color="red", label="before", alpha=0.4)
                for l in y2:
                    plt.plot(l, color="blue", label="after", alpha=0.4)
                plt.legend()
                pub.save_fig("output/task" + str(task_number) + ".png")
                plt.close()
                number_printed += 1
        task_number += 1
    pbar.close()
    avg_solution_before /= len(json_before["tasks"])
    avg_solution_after /= len(json_after["tasks"])
    print(
        "Average number of programs enumerated before finding a solution:\n\t- before:",
        avg_solution_before,
        "\n\t- after:",
        avg_solution_after,
    )
    print(
        "Ratio avg before/after (in %):\n\t",
        avg_solution_before / avg_solution_after * 100,
    )
    print(
        "Number of timeouts:\n\t- before:",
        len(timeout_before),
        "\n\t- after:",
        len(timeout_after),
    )
    print(
        "List of tasks that timeout:\n\t- before:",
        timeout_before,
        "\n\t- after:",
        timeout_after,
    )
    if len(timeout_after) != 0:
        print("Ratio timeout before/after:\t", len(timeout_before) / len(timeout_after))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two models")

    parser.add_argument(
        "-jsonb",
        "--json-before",
        type=str,
        default=False,
        help="JSON file of analysis done in state 'before'",
    )

    parser.add_argument(
        "-jsona",
        "--json-after",
        type=str,
        default=False,
        help="JSON file of analysis done in state 'after'",
    )

    parser.add_argument(
        "-nbtp",
        "--number-task-print",
        type=int,
        default=30,
        help="Number of printed tasks",
    )

    parameters = parser.parse_args()
    json_filename_before: str = parameters.json_before
    json_filename_after: str = parameters.json_after
    number_task_print: int = parameters.number_task_print
    json_before = load_json(json_filename_before)
    json_after = load_json(json_filename_after)
    MAX_TESTS: int = max(json_before["max_tests"], json_after["max_tests"])

    compare_jsons(json_before, json_after)
