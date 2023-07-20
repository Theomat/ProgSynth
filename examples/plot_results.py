from glob import glob
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub
import csv
import argparse

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
    "--no-programs",
    action="store_true",
    default=False,
    help="does not show programs wrt tasks",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="verbose mode",
)
parameters = parser.parse_args()
dataset_file: str = parameters.dataset
output_folder: str = parameters.folder
no_show: bool = parameters.no_show
no_sort: bool = parameters.no_sort
no_progs: bool = parameters.no_programs
verbose: bool = parameters.verbose

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

pub.setup()


def ready_ca(x_axis_label: str) -> None:
    """
    Make the current axe ready
    """
    plt.xlabel(x_axis_label)
    plt.ylabel("Tasks Completed")
    plt.grid()


def update_limits(ax: plt.Axes, max_x: int, max_y: int) -> None:
    """
    Makes a dashed line for the max number of tasks and update each axis' limits.
    """
    ax.hlines(
        [max_y],
        xmin=0,
        xmax=max_x,
        label="All tasks",
        color="k",
        linestyles="dashed",
    )
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y + 10)


ax1 = plt.subplot(1, 1 + int(not no_progs), 1)
ready_ca("Time (in s)")
if not no_progs:
    ax2 = plt.subplot(1, 2, 2)
    ready_ca("# Programs")

max_time, max_programs = 0, 0
max_tasks = 0

# Dict[name, Dict[seed, data]]
methods = {}

for file in glob(os.path.join(output_folder, "*.csv")):
    filename = os.path.relpath(file, output_folder)
    if verbose:
        print("found csv file:", file)
    # filename should be {dataset_name}_seed_{seed}_{name}.csv
    if not filename.startswith(dataset_name):
        if verbose:
            print(f"\tskipped: does not start with {dataset_name}")
        continue
    name = filename[len(dataset_name) : -4]
    if not name.startswith("_seed_"):
        if verbose:
            print(f"\tskipped: does not start with {dataset_name}_seed_")
        continue
    name = name[6:]
    seed = int(name[: name.index("_")])
    name = name[name.index("_") + 1 :].replace("_", " ")
    if name not in methods:
        methods[name] = {}
    if seed in methods[name]:
        print(f"Warning: duplicate seed {seed} for method {name}!")
        continue
    # Load data from the file
    trace = []
    with open(os.path.join(output_folder, filename), "r") as fd:
        reader = csv.reader(fd)
        trace = [tuple(row) for row in reader]
        # Pop columns names
        trace.pop(0)
        # Type conversion (success, time, num_of_programs)
        trace = [(row[0] == "True", float(row[1]), int(row[2])) for row in trace]
    if len(trace) == 0:
        if verbose:
            print(f"\tskipped: no data")
        continue
    # Compute data for first plot
    trace_time = trace if no_sort else sorted(trace, key=lambda x: x[1])
    cum_sol1, cum_time = np.cumsum([row[0] for row in trace_time]), np.cumsum(
        [row[1] for row in trace_time]
    )
    # Compute data for second plot
    trace_programs = trace if no_sort else sorted(trace, key=lambda x: x[2])
    cum_sol2, cum_programs = np.cumsum([row[0] for row in trace_programs]), np.cumsum(
        [row[2] for row in trace_programs]
    )
    # Save data for method
    methods[name][seed] = (cum_sol1, cum_time, cum_sol2, cum_programs)
    # Update maxis
    max_tasks = max(len(trace), max_tasks)
    max_time = max(max_time, cum_time[-1])
    max_programs = max(max_programs, cum_programs[-1])

    print(f"{name} (seed={seed}) solved", cum_sol2[-1], "/", len(trace))

if verbose:
    print(f"Max Time: {max_time}s")
    print(f"Max Programs: {max_programs}")


def plot_with_incertitude(
    ax: plt.Axes, x: List[np.ndarray], y: List[np.ndarray], label: str
) -> None:
    max_len = max(len(xi) for xi in x)
    x = [xi for xi in x if len(xi) == max_len]
    y = [yi for yi in y if len(yi) == max_len]

    x_min = np.min(np.array(x))
    x_max = np.max(np.array(x))
    target_x = np.arange(x_min, x_max + 1, step=(x_max - x_min) / 50)
    # Interpolate data
    data = []
    for xi, yi in zip(x, y):
        nyi = np.interp(target_x, xi, yi)
        data.append(nyi)
    # Compute distribution
    Y = np.array(data)
    mean = np.mean(Y, axis=0)
    std = 1.96 * np.std(Y, axis=0)

    p = ax.plot(target_x, mean, label=label)
    color = p[0].get_color()
    ax.fill_between(target_x, mean - std, mean + std, color=color, alpha=0.5)


# Check we have at least one file
if len(methods) == 0:
    import sys

    print("Error: no performance file was found!", file=sys.stderr)
    sys.exit(1)

# Plot data with incertitude
for method, seeds_dico in methods.items():
    seeds = list(seeds_dico.keys())
    plot_with_incertitude(
        ax1,
        [seeds_dico[seed][1] for seed in seeds],
        [seeds_dico[seed][0] for seed in seeds],
        method.capitalize(),
    )
    if not no_progs:
        plot_with_incertitude(
            ax2,
            [seeds_dico[seed][3] for seed in seeds],
            [seeds_dico[seed][2] for seed in seeds],
            method.capitalize(),
        )


# Update limits and legendss
update_limits(ax1, max_time, max_tasks)
ax1.legend()
if not no_progs:
    update_limits(ax2, max_programs, max_tasks)
    ax2.legend()
pub.save_fig(os.path.join(output_folder, "results.png"))
if not no_show:
    plt.show()
