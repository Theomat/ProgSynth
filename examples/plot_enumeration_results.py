from glob import glob
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import pltpublish as pub
import csv

from plot_helper import (
    plot_y_wrt_x,
    make_plot_wrapper,
    plot_dist,
    plot_rank_by,
)


__DATA__ = {
    "time": (0, "Time (in s)", 5, False, False),
    "programs": (1, "Programs Enumerated", 0, False, False),
    "queued": (2, "Programs Enqueued", 0, False, False),
    "banked": (3, "Programs in Banks", 0, False, False),
    "non_terminals": (4, "Non Terminals in Grammar", 0, False, False),
}


def load_data(output_folder: str, verbose: bool = False) -> Dict[str, Dict[int, List]]:
    # Dict[name, Dict[seed, data]]
    methods = {}

    chosen_dsl = None

    for file in glob(os.path.join(output_folder, "*.csv")):
        filename = os.path.relpath(file, output_folder)
        if verbose:
            print("found csv file:", file)
        # filename should end with dsl_{dsl}_seed_{seed}_depth_{max_depth}.csv
        name = filename[:-4]
        if "_seed_" not in name:
            if verbose:
                print(f"\tskipped: does not contain _seed_")
            continue

        dsl = name[name.index("dsl_") + 4 : name.index("_seed_")].replace("_", " ")
        if not (chosen_dsl is None or dsl == chosen_dsl):
            if verbose:
                print(
                    f"\tskipped: does not contain data belonging to the same dsl: {dsl}"
                )
                continue
            if chosen_dsl is None:
                chosen_dsl = dsl
                if verbose:
                    print(f"found dsl: {dsl}")
        seed = int(name[name.index("_seed_") + 6 : name.index("_depth_")])

        trace = []
        with open(os.path.join(output_folder, filename), "r") as fd:
            reader = csv.reader(fd)
            trace = [tuple(row) for row in reader]
            # Pop columns names
            columns = {name: ind for ind, name in enumerate(trace.pop(0))}
            indices = [
                columns["search"],
                columns["time"],
                columns["programs"],
                columns["queue"],
                columns["bank"],
                columns.get("non_terminals", -1),
            ]
            data = [tuple(row[k] if k >= 0 else 0 for k in indices) for row in trace]
            if len(data) == 0:
                if verbose:
                    print(f"\tskipped: no data")
                continue
            agg = defaultdict(list)
            for row in data:
                agg[row[0]].append(row[1:])

            for name, data in agg.items():
                if name not in methods:
                    methods[name] = {}
                if seed in methods[name]:
                    print(f"Warning: duplicate seed {seed} for method {name}!")
                    continue

                # Save data for method
                methods[name][seed] = [tuple(float(x) for x in row) for row in data]
    return methods


# Generate all possible combinations
__PLOTS__ = {}
for ydata in list(__DATA__.keys()):
    for xdata in list(__DATA__.keys()):
        if xdata == ydata:
            continue
        if xdata == "non_terminals":
            __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_wrapper(
                plot_y_wrt_x,
                __DATA__[xdata],
                __DATA__[ydata],
                cumulative=False,
                logy=True,
            )
        else:
            __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_wrapper(
                plot_y_wrt_x, __DATA__[xdata], __DATA__[ydata], cumulative=False
            )
    if ydata != "tasks":
        __PLOTS__[f"rank_by_{ydata}"] = make_plot_wrapper(plot_rank_by, __DATA__[ydata])
        __PLOTS__[f"dist_{ydata}_by_programs"] = make_plot_wrapper(
            plot_dist, __DATA__[ydata], "tasks"
        )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Plot results")

    parser.add_argument(
        "--folder",
        type=str,
        default="./",
        help="folder in which to look for CSV files (default: './')",
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
    output_folder: str = parameters.folder
    verbose: bool = parameters.verbose
    plots: List[str] = parameters.plots

    # Load data
    pub.setup()
    methods = load_data(output_folder, verbose)
    # Check we have at least one file
    if len(methods) == 0:
        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)
    # Order by name so that it is always the same color for the same methods if diff. DSL
    ordered_methods = OrderedDict()
    for met in sorted(methods.keys()):
        ordered_methods[met] = methods[met]
    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, ordered_methods)
    plt.show()
