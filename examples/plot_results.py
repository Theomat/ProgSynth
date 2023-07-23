from glob import glob
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub
import csv


def load_data(
    dataset_name: str, output_folder: str, verbose: bool = False
) -> Tuple[Dict[str, Dict[int, List]], float]:
    # Dict[name, Dict[seed, data]]
    methods = {}
    timeout = 1e99

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
            for x in trace:
                if x[0] == 0:
                    timeout = min(x[1], timeout)
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
    return methods, timeout


def make_filter_wrapper(func, *args) -> None:
    def f(task_index: int, methods: Dict[str, Dict[int, List]], timeout: float) -> bool:
        return func(task_index, methods, timeout, *args)

    return f


def timeout_filter(
    task_index: int,
    methods: Dict[str, Dict[int, List]],
    timeout: float,
    nbr_timeouts: int,
) -> bool:
    timeouts = 0
    for method, seeds_dico in methods.items():
        if nbr_timeouts == -1:
            nbr_timeouts = len(methods) * len(seeds_dico)
        for seed, data in seeds_dico.items():
            timeouts += 1 - data[task_index][0]
            if timeouts > nbr_timeouts:
                return False

    return True


def time_filter(
    task_index: int,
    methods: Dict[str, Dict[int, List]],
    timeout: float,
    ratio: float,
    aggregator,
) -> bool:
    all_times = []
    for method, seeds_dico in methods.items():
        for seed, data in seeds_dico.items():
            all_times.append(data[task_index][1])
    return aggregator(all_times) >= ratio * timeout


def reverse_filter(func):
    def f(
        task_index: int, methods: Dict[str, Dict[int, List]], timeout: float, *args
    ) -> bool:
        return not func(task_index, methods, timeout, *args)

    return f


__FILTERS__ = {
    "timeouts.none": make_filter_wrapper(timeout_filter, 1),
    "solve>=1": make_filter_wrapper(timeout_filter, -1),
}

for ratio in [0.25, 0.5, 0.75]:
    for name, aggr in [
        ("fastest", np.min),
        ("mean", np.mean),
        ("median", np.median),
        ("slowest", np.max),
    ]:
        __FILTERS__[f"time.{name}>={ratio:.0%}"] = make_filter_wrapper(
            time_filter, ratio, aggr
        )

for key in list(__FILTERS__.keys()):
    __FILTERS__[f"not.{key}"] = reverse_filter(__FILTERS__[key])


def filter(
    methods: Dict[str, Dict[int, List]], filter_name: str, timeout: float
) -> Dict[str, Dict[int, List]]:
    fun = __FILTERS__[filter_name]
    task_len = len(list(list(methods.values())[0].values())[0])
    should_keep = [fun(i, methods, timeout) for i in range(task_len)]
    return {
        m: {
            s: [x for i, x in enumerate(data) if should_keep[i]]
            for s, data in val.items()
        }
        for m, val in methods.items()
    }


def plot_with_incertitude(
    ax: plt.Axes,
    x: List[np.ndarray],
    y: List[np.ndarray],
    label: str,
    std_factor: float = 1.96,
    miny: Optional[float] = None,
    maxy: Optional[float] = None,
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
    upper = mean + std
    if maxy is not None:
        upper = np.minimum(upper, maxy)
    lower = mean - std
    if miny is not None:
        lower = np.maximum(lower, miny)
    ax.fill_between(target_x, lower, upper, color=color, alpha=0.5)


__DATA__ = {
    "tasks": (0, "Tasks completed", 10, True, True),
    "time": (1, "Time (in s)", 5, False, False),
    "programs": (2, "Programs Enumerated", 0, False, False),
}


def plot_y_wrt_x(
    ax: plt.Axes,
    methods: Dict[str, Dict[int, List]],
    should_sort: bool,
    x_name: str,
    y_name: str,
) -> None:
    # Plot data with incertitude
    a_index, a_name, a_margin, show_len_a, _ = __DATA__[y_name]
    b_index, b_name, b_margin, show_len_b, _ = __DATA__[x_name]
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
        max_a = max(max(np.max(yi) for yi in ydata), max_a)
        max_b = max(max(np.max(xi) for xi in xdata), max_b)
        plot_with_incertitude(
            ax,
            xdata,
            ydata,
            method.capitalize(),
            maxy=data_length if show_len_a else None,
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


def make_plot_wrapper(func, *args) -> None:
    def f(ax: plt.Axes, methods: Dict[str, Dict[int, List]], should_sort: bool) -> None:
        return func(ax, methods, should_sort, *args)

    return f


def get_rank_matrix(
    methods: Dict[str, Dict[int, List]], yindex: int, maximize: bool
) -> Tuple[List[str], np.ndarray]:
    seeds = list(methods.values())[0].keys()
    task_len = len(list(list(methods.values())[0].values())[0])
    rank_matrix = np.ndarray((len(methods), task_len, len(methods)), dtype=float)
    method_names = list(methods.keys())
    data = np.ndarray((len(methods), len(seeds)), dtype=float)
    np.random.seed(1)
    for task_no in range(task_len):
        for i, method in enumerate(method_names):
            for j, seed in enumerate(seeds):
                data[i, j] = methods[method][seed][task_no][yindex]
        # data_for_seed = []
        # for method in method_names:
        #     data = methods[method][seed]
        #     data_for_seed.append([d[yindex] for d in data])
        # data_for_seed = np.array(data_for_seed)
        if maximize:
            data = -data
        rand_x = np.random.random(size=data.shape)
        # This is done to randomly break ties.
        # Last key is the primary key,
        indices = np.lexsort((rand_x, data), axis=0)
        for i, method in enumerate(method_names):
            rank_matrix[i, task_no] = [
                np.sum(indices[i] == rank) / len(seeds) for rank in range(len(methods))
            ]
    return rank_matrix


def __ready_for_stacked_dist_plot__(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=True,
        right=False,
        labeltop=False,
        labelbottom=True,
        labelleft=True,
        labelright=False,
    )

    ax.legend(fancybox=True, fontsize="large")


def plot_rank_by(
    ax: plt.Axes, methods: Dict[str, Dict[int, List]], should_sort: bool, y_name: str
) -> None:
    width = 1.0
    a_index, a_name, a_margin, show_len_a, should_max = __DATA__[y_name]
    rank_matrix = get_rank_matrix(methods, a_index, should_max)
    labels = list(range(1, len(methods) + 1))
    mean_ranks = np.mean(rank_matrix, axis=-2)
    bottom = np.zeros_like(mean_ranks[0])
    for i, key in enumerate(methods.keys()):
        label = key
        bars = ax.bar(
            labels,
            mean_ranks[i],
            width,
            label=label,
            bottom=bottom,
            alpha=0.9,
            linewidth=1,
            edgecolor="white",
        )
        ax.bar_label(bars, labels=[f"{x:.1%}" for x in mean_ranks[i]])
        bottom += mean_ranks[i]

    ax.set_ylabel("Fraction (in %)", size="large")
    yticks = np.array(range(0, 101, 20))
    ax.set_yticklabels(yticks)
    ax.set_yticks(yticks * 0.01)
    ax.set_xlabel("Ranking", size="large")
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    __ready_for_stacked_dist_plot__(ax)
    word = "Most" if should_max else "Least"
    ax.set_title(f"{word} {a_name}")


def plot_dist(
    ax: plt.Axes, methods: Dict[str, Dict[int, List]], should_sort: bool, y_name: str
) -> None:
    width = 1.0
    data_length = 0
    a_index, a_name, a_margin, show_len_a, should_max = __DATA__[y_name]
    max_a = max(
        max(max([y[a_index] for y in x]) for x in seed_dico.values())
        for seed_dico in methods.values()
    )
    bottom = None
    nbins = 5
    bins = [max_a]
    while len(bins) <= nbins:
        bins.insert(0, np.sqrt(bins[0] + 1))
    for i in range(nbins):
        if bins[i + 1] < 2 * bins[i]:
            bins[i + 1] = 2 * bins[i]
    x_bar = list(range(1, nbins + 1))
    for method, seeds_dico in methods.items():
        hists = []
        for seed, raw_data in seeds_dico.items():
            data = [x[a_index] for x in raw_data]
            data_length = max(data_length, len(data))
            hist, edges = np.histogram(
                data, bins=bins, range=(1e-3, max_a), density=False
            )
            hists.append(hist)
        true_hist = np.mean(hists, axis=0) / data_length
        if bottom is None:
            bottom = np.zeros_like(true_hist)
        label = method
        bars = ax.bar(
            x_bar,
            true_hist,
            width,
            label=label,
            bottom=bottom,
            alpha=0.9,
            linewidth=1,
            edgecolor="white",
        )
        ax.bar_label(bars, labels=[f"{x:.1%}" for x in true_hist])
        bottom += true_hist
    __ready_for_stacked_dist_plot__(ax)
    ax.set_yticklabels([])
    ax.set_xlabel(a_name, size="large")
    ax.set_xticklabels(map(lambda x: f"<{x:.0f}", edges))
    ax.set_title(f"Distribution of {a_name} per task")


# Generate all possible combinations
__PLOTS__ = {}
for ydata in list(__DATA__.keys()):
    for xdata in list(__DATA__.keys()):
        if xdata == ydata:
            continue
        __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_wrapper(
            plot_y_wrt_x, xdata, ydata
        )
    if ydata != "tasks":
        __PLOTS__[f"rank_by_{ydata}"] = make_plot_wrapper(plot_rank_by, ydata)
        __PLOTS__[f"dist_{ydata}_by_task"] = make_plot_wrapper(plot_dist, ydata)


if __name__ == "__main__":
    import argparse
    import sys

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
    parser.add_argument(
        "--filter",
        type=str,
        nargs="*",
        choices=list(__FILTERS__.keys()),
        help="filter tasks (keep data based on the filter)",
    )
    parser.add_argument("plots", nargs="+", choices=list(__PLOTS__.keys()))
    parameters = parser.parse_args()
    dataset_file: str = parameters.dataset
    output_folder: str = parameters.folder
    verbose: bool = parameters.verbose
    no_sort: bool = not parameters.sorted
    plots: List[str] = parameters.plots
    filters: List[str] = parameters.filter or []

    # Initial Setup
    start_index = (
        0
        if not os.path.sep in dataset_file
        else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
    )
    dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]
    # Load data
    pub.setup()
    methods, timeout = load_data(dataset_name, output_folder, verbose)
    # Check we have at least one file
    if len(methods) == 0:
        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)
    for filter_name in filters:
        methods = filter(methods, filter_name, timeout)
        # Check we did not remove everything
        task_len = len(list(list(methods.values())[0].values())[0])
        if task_len == 0:

            print("Error: filters left no tasks!", file=sys.stderr)
            sys.exit(1)

    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, methods, not no_sort)
    plt.show()
