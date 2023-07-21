from glob import glob
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub
import csv


def load_data(no_sort: bool):
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
        cum_sol2, cum_programs = np.cumsum(
            [row[0] for row in trace_programs]
        ), np.cumsum([row[2] for row in trace_programs])
        # Save data for method
        methods[name][seed] = (cum_sol1, cum_time, cum_sol2, cum_programs)
        # Update maxis
        max_tasks = max(len(trace), max_tasks)
        max_time = max(max_time, cum_time[-1])
        max_programs = max(max_programs, cum_programs[-1])

        print(f"{name} (seed={seed}) solved", cum_sol2[-1], "/", len(trace))
    return methods, max_tasks, max_programs, max_time


def plot_with_incertitude(
    ax: plt.Axes,
    x: List[np.ndarray],
    y: List[np.ndarray],
    label: str,
    std_factor: float = 1.96,
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
    std = std_factor * np.std(Y, axis=0)

    p = ax.plot(target_x, mean, label=label)
    color = p[0].get_color()
    ax.fill_between(target_x, mean - std, mean + std, color=color, alpha=0.5)


def plot_tasks_wrt_time(
    ax: plt.Axes, methods: Dict[str, Dict[int, List[float]]]
) -> None:
    # Plot data with incertitude
    for method, seeds_dico in methods.items():
        seeds = list(seeds_dico.keys())
        plot_with_incertitude(
            ax,
            [seeds_dico[seed][1] for seed in seeds],
            [seeds_dico[seed][0] for seed in seeds],
            method.capitalize(),
        )
    ax.set_xlabel("Time (in s)")
    ax.set_ylabel("Tasks Completed")
    ax.grid()
    ax.hlines(
        [max_tasks],
        xmin=0,
        xmax=max_time,
        label="All tasks",
        color="k",
        linestyles="dashed",
    )
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_tasks + 10)


def plot_tasks_wrt_programs(
    ax: plt.Axes, methods: Dict[str, Dict[int, List[float]]]
) -> None:
    # Plot data with incertitude
    for method, seeds_dico in methods.items():
        seeds = list(seeds_dico.keys())
        plot_with_incertitude(
            ax,
            [seeds_dico[seed][3] for seed in seeds],
            [seeds_dico[seed][2] for seed in seeds],
            method.capitalize(),
        )
    ax.set_xlabel("Programs enumerated")
    ax.set_ylabel("Tasks Completed")
    ax.grid()
    ax.hlines(
        [max_tasks],
        xmin=0,
        xmax=max_programs,
        label="All tasks",
        color="k",
        linestyles="dashed",
    )
    ax.set_xlim(0, max_programs)
    ax.set_ylim(0, max_tasks + 10)


__PLOTS__ = {
    "tasks_wrt_time": plot_tasks_wrt_time,
    "tasks_wrt_programs": plot_tasks_wrt_programs,
}

if __name__ == "__main__":
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
        "--sorted", action="store_true", help="sort data by task solving time"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose mode",
    )
    parser.add_argument("plots", nargs="+", choices=list(__PLOTS__.keys()))
    parameters = parser.parse_args()
    dataset_file: str = parameters.dataset
    output_folder: str = parameters.folder
    verbose: bool = parameters.verbose
    no_sort: bool = not parameters.sorted
    plots: List[str] = parameters.plots

    # Initial Setup
    start_index = (
        0
        if not os.path.sep in dataset_file
        else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
    )
    dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]
    # Load data
    pub.setup()
    methods, max_tasks, max_programs, max_time = load_data(no_sort)

    if verbose:
        print(f"Max Time: {max_time}s")
        print(f"Max Programs: {max_programs}")
    # Check we have at least one file
    if len(methods) == 0:
        import sys

        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)

    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, methods)

    plt.show()
