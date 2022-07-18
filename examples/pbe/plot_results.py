from glob import glob
import os
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


parameters = parser.parse_args()
dataset_file: str = parameters.dataset
output_folder: str = parameters.folder

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

pub.setup()
ax1 = plt.subplot(1, 2, 1)
plt.xlabel("Time (in s)")
plt.ylabel("Tasks Completed")
plt.grid()
ax2 = plt.subplot(1, 2, 2)
plt.xlabel("# Programs")
plt.ylabel("Tasks Completed")
plt.grid()
max_time, max_programs = 0, 0
max_tasks = 0
for file in glob(os.path.join(output_folder, "*.csv")):
    file = os.path.relpath(file, output_folder)
    if not file.startswith(dataset_name):
        continue
    name = file[len(dataset_name) : -4]
    if "_" not in name:
        continue
    name = name[name.index("_") + 1 :].replace("_", " ")
    trace = []
    with open(file, "r") as fd:
        reader = csv.reader(fd)
        trace = [tuple(row) for row in reader]
        trace.pop(0)
        trace = [(row[0] == "True", float(row[1]), int(row[2])) for row in trace]
        max_tasks = max(len(trace), max_tasks)
    # Plot tasks wrt time
    trace_time = sorted(trace, key=lambda x: x[1])
    cum_sol1, cum_time = np.cumsum([row[0] for row in trace_time]), np.cumsum(
        [row[1] for row in trace_time]
    )
    max_time = max(max_time, cum_time[-1])
    ax1.plot(cum_time, cum_sol1, label=name.capitalize())
    # Plot tasks wrt programs
    trace_programs = sorted(trace, key=lambda x: x[2])
    cum_sol2, cum_programs = np.cumsum([row[0] for row in trace_programs]), np.cumsum(
        [row[2] for row in trace_programs]
    )
    max_programs = max(max_programs, cum_programs[-1])
    ax2.plot(cum_programs, cum_sol2, label=name.capitalize())
    print(name, "solved", cum_sol2[-1], "/", len(trace))
ax1.hlines(
    [max_tasks],
    xmin=0,
    xmax=max_time,
    label="All tasks",
    color="k",
    linestyles="dashed",
)
ax1.set_xlim(0, max_time)
ax1.set_ylim(0, max_tasks + 10)
ax2.hlines(
    [max_tasks],
    xmin=0,
    xmax=max_programs,
    label="All tasks",
    color="k",
    linestyles="dashed",
)
ax2.set_xlim(0, max_programs)
ax2.set_ylim(0, max_tasks + 10)
ax1.legend()
ax2.legend()
pub.save_fig(os.path.join(output_folder, "results.png"))
plt.show()
