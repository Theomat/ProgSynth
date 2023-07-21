from glob import glob
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub
import csv


def load_data(
    dataset_name: str, output_folder: str, verbose: bool = False
) -> Dict[str, Dict[int, List]]:
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
            trace = [
                (int(row[0] == "True"), float(row[1]), int(row[2])) for row in trace
            ]
        if len(trace) == 0:
            if verbose:
                print(f"\tskipped: no data")
            continue
        # Save data for method
        methods[name][seed] = trace
        if verbose:
            print(
                f"{name} (seed={seed}) solved",
                sum(x[0] for x in trace),
                "/",
                len(trace),
            )
    return methods


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


__DATA__ = {
    "tasks": (0, "Tasks completed", 10, True),
    "time": (1, "Time (in s)", 5, False),
    "programs": (2, "Programs Enumerated", 0, False),
}


def plot_y_wrt_x(
    ax: plt.Axes,
    methods: Dict[str, Dict[int, List]],
    should_sort: bool,
    x_name: str,
    y_name: str,
) -> None:
    # Plot data with incertitude
    a_index, a_name, a_margin, show_len_a = __DATA__[y_name]
    b_index, b_name, b_margin, show_len_b = __DATA__[x_name]
    max_a = 0
    max_b = 0
    data_length = 0
    for method, seeds_dico in methods.items():
        seeds = list(seeds_dico.keys())
        data = [
            [(elems[b_index], elems[a_index]) for elems in seeds_dico[seed]]
            for seed in seeds
        ]
        data_length = max(data_length, len(data[0]))
        if should_sort:
            data = [sorted(seed_data) for seed_data in data]
        xdata = [np.cumsum([x[0] for x in seed_data]) for seed_data in data]
        ydata = [np.cumsum([x[1] for x in seed_data]) for seed_data in data]
        max_a = max(np.max(ydata), max_a)
        max_b = max(np.max(xdata), max_b)
        plot_with_incertitude(
            ax,
            xdata,
            ydata,
            method.capitalize(),
        )
    ax.set_xlabel(b_name)
    ax.set_ylabel(a_name)
    ax.grid()
    if show_len_a:
        ax.hlines(
            [data_length],
            xmin=0,
            xmax=max_b + b_margin,
            label=f"All {y_name}",
            color="k",
            linestyles="dashed",
        )
        max_a = data_length
    ax.set_xlim(0, max_b + b_margin)
    ax.set_ylim(0, max_a + a_margin)
    ax.legend()


def make_plot_builder_for(x_name: str, y_name: str) -> None:
    def f(ax: plt.Axes, methods: Dict[str, Dict[int, List]], should_sort: bool) -> None:
        return plot_y_wrt_x(ax, methods, should_sort, x_name, y_name)

    return f


__PLOTS__ = {}
for ydata in list(__DATA__.keys()):
    for xdata in list(__DATA__.keys()):
        if xdata == ydata:
            continue
        __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_builder_for(xdata, ydata)

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
    methods = load_data(dataset_name, output_folder, verbose)

    # Check we have at least one file
    if len(methods) == 0:
        import sys

        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)

    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, methods, not no_sort)

    plt.show()
