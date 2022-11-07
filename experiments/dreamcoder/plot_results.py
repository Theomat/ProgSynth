from glob import glob
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub
import argparse


from synth.task import Dataset

parser = argparse.ArgumentParser(description="Plot results")
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="dataset.pickle",
    help="dataset (default: dataset.pickle)",
)
parser.add_argument(
    "--folder",
    type=str,
    default="./",
    help="folder in which to look for CSV files (default: './')",
)
parser.add_argument(
    "--no-show",
    action="store_true",
    default=False,
    help="just save the image does not show it",
)
parser.add_argument(
    "--no-sort",
    action="store_true",
    default=False,
    help="does not sort tasks by time taken",
)
parser.add_argument(
    "--support",
    type=str,
    default=None,
    help="train dataset to get the set of supported type requests",
)
parser.add_argument(
    "--no-failure",
    action="store_true",
    default=False,
    help="does not failed tasks",
)
parameters = parser.parse_args()
dataset_file: str = parameters.dataset
output_folder: str = parameters.folder
no_show: bool = parameters.no_show
no_sort: bool = parameters.no_sort
support: str = parameters.support.format(dsl_name="dreamcoder")
no_failure: bool = parameters.no_failure


supported_type_requests = Dataset.load(support).type_requests() if support else None
start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

pub.setup()


def read_csv(
    file: str, separator: str = ",", dump_first_line: bool = True
) -> List[List[str]]:
    out = []
    print(file)
    with open(file, "r") as fd:
        lines = fd.readlines()
        if dump_first_line:
            lines.pop(0)
        out = [line.split(separator) for line in lines]
    return out


max_time, max_programs = 0, 0
max_tasks = 0
mask = []
tasks = [
    task
    for task in Dataset.load(dataset_file).tasks
    if supported_type_requests is None or task.type_request in supported_type_requests
]
exact = ["repeat"]
forbidden = [
    "count",
    "kth",
    "remove-mod",
    "replace",
    "repeat-k with k=2",
    "repeat-k with k=3",
    "repeat-k with k=4",
    "repeat-k with k=5",
    "take-k with k=3",
    "take-k with k=4",
    "take-k with k=5",
    "reverse",
    "drop-k with k=4",
    "drop-k with k=5",
    "dup",
    "fibonacci",
    "max",
    "min",
    "product",
    "sort",
    "sum",
    "remove-index-k with k=2",
    "remove-index-k with k=3",
    "remove-index-k with k=4",
    "remove-index-k with k=5",
    "rotate",
    "odds",
    "pop",
    "pow-k with k=2",
    "pow-k with k=3",
    "pow-k with k=4",
    "pow-k with k=5",
    "slice-k-n with k=1 and n=3",
    "slice-k-n with k=1 and n=4",
    "slice-k-n with k=1 and n=5",
    "slice-k-n with k=3 and n=2",
    "slice-k-n with k=3 and n=3",
    "slice-k-n with k=3 and n=4",
    "slice-k-n with k=3 and n=5",
    "slice-k-n with k=4 and n=2",
    "slice-k-n with k=4 and n=3",
    "slice-k-n with k=4 and n=4",
    "slice-k-n with k=4 and n=5",
    "slice-k-n with k=5 and n=2",
    "slice-k-n with k=5 and n=3",
    "slice-k-n with k=5 and n=4",
    "slice-k-n with k=5 and n=5",
    "slice-k-n with k=2 and n=3",
    "slice-k-n with k=2 and n=4",
    "slice-k-n with k=2 and n=5",
]
for i, task in enumerate(tasks):
    name = task.metadata["name"]
    mask.append(
        (not any(name.startswith(prefix) for prefix in forbidden)) and not name in exact
    )


to_plot = []
for file in glob(os.path.join(output_folder, "*.csv")):
    file = os.path.relpath(file, output_folder)
    if not file.startswith(dataset_name):
        continue
    name = file[len(dataset_name) : -4]
    if "_" not in name:
        continue
    name = name[name.index("_") + 1 :].replace("_", " ")
    if "ucfg" in name:
        name = "U-CFG"
    else:
        name = "det-CFG"
    trace = []
    original = read_csv(os.path.join(output_folder, file))
    trace = [(row[0] == "True", float(row[1]), int(row[2])) for row in original]
    j = 0
    for task, result in zip(tasks, trace):
        if result[0] and not mask[j]:
            print(task.metadata["name"], original[j])
            # print("We solved it even though it's forbidden:", task.metadata["name"])
        j += 1
    trace = [row for i, row in enumerate(trace) if mask[i]]
    max_tasks = max(len(trace), max_tasks)

    to_plot.append((name, trace))


solved_by_any = [any(t[i][0] for _, t in to_plot) for i in range(max_tasks)]
mini = [
    min(t[i][2] for _, t in to_plot)
    for i in range(max_tasks)
    if solved_by_any[i] or not no_failure
]

plt.figure()
plt.xlabel("Time (in s)")
plt.ylabel("Tasks Completed")
plt.grid()

for name, trace in to_plot:
    trace = [row for i, row in enumerate(trace) if solved_by_any[i] or not no_failure]
    # Plot tasks wrt time
    trace_time = trace if no_sort else sorted(trace, key=lambda x: x[1])
    cum_sol1, cum_time = np.cumsum([row[0] for row in trace_time]), np.cumsum(
        [row[1] for row in trace_time]
    )
    max_time = max(max_time, cum_time[-1])
    plt.plot(cum_time, cum_sol1, label=name)
    print(name, "solved", cum_sol1[-1], "/", len(trace))
    if no_failure:
        max_tasks = len(trace)


plt.hlines(
    [max_tasks],
    xmin=0,
    xmax=max_time,
    label="All solved tasks" if no_failure else "All tasks",
    color="k",
    linestyles="dashed",
)
plt.xlim(0, max_time)
plt.ylim(0, max_tasks + 10)

pub.save_fig(os.path.join(output_folder, "time.png"))
if not no_show:
    plt.show()
plt.figure()
plt.xlabel("# Programs")
plt.ylabel("Tasks Completed")
plt.grid()
for name, trace in to_plot:
    trace = [row for i, row in enumerate(trace) if solved_by_any[i] or not no_failure]
    # Plot tasks wrt programs
    trace_programs = trace if no_sort else sorted(trace, key=lambda x: x[2])
    cum_sol2, cum_programs = np.cumsum([row[0] for row in trace_programs]), np.cumsum(
        [row[2] for row in trace_programs]
    )
    max_programs = max(max_programs, cum_programs[-1])
    plt.plot(cum_programs, cum_sol2, label=name)
plt.hlines(
    [max_tasks],
    xmin=0,
    xmax=max_programs,
    label="All solved tasks" if no_failure else "All tasks",
    color="k",
    linestyles="dashed",
)
plt.xlim(0, max_programs)
plt.ylim(0, max_tasks + 10)
plt.legend()
pub.save_fig(os.path.join(output_folder, "programs.png"))
if not no_show:
    plt.show()

plt.figure()
all_data = []
labels = [name for name, _ in to_plot]
for name, trace in to_plot:
    trace = [row for i, row in enumerate(trace) if solved_by_any[i] or not no_failure]
    all_data.append(np.array([1 + a[2] - mi for a, mi in zip(trace, mini)]))
    # plt.plot(, label=name, linestyle='None', markersize = 10.0, marker='x')
bplot = plt.boxplot(
    all_data,
    patch_artist=True,  # vertical box alignment)  # fill with color)
    showmeans=True,  # vertical box alignment)  # fill with color)
    vert=True,
)  # vertical box alignment)  # fill with color)
colors = ["lightblue", "lightgreen"]
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)

# bplot = plt.violinplot(all_data,
#             showmeans=True,  # vertical box alignment)  # fill with color)
#             vert=True,)  # vertical box alignment)  # fill with color)

plt.gca().set_xticks([y + 1 for y in range(len(all_data))], labels=labels)
plt.grid(True)
plt.yscale("log", base=10)
# plt.xlim(0, max_tasks)
plt.ylim(1, np.max(all_data))
plt.xlabel("Model")
plt.ylabel(
    "Distribution of Number of Programs Enumerated after best model found a solution"
)
plt.legend()
plt.show()