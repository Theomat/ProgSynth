from argparse import ArgumentParser
import pickle

from colorama import Fore as F

from synth import Dataset
from synth.utils import chrono


class DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except:
            return super().find_class(module + "." + module, name)


def add_dataset_choice_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        default="{dsl_name}.pickle",
        help="the dataset file to load (default: {dsl_name}.pickle)",
    )


def load_dataset(dsl_name: str, dataset_file: str, verbose: bool = True) -> Dataset:
    dataset_file = dataset_file.format(dsl_name=dsl_name)
    if verbose:
        print(f"Loading {F.LIGHTCYAN_EX}{dataset_file}{F.RESET}...", end="")
    with chrono.clock("dataset.load") as c:
        full_dataset: Dataset = Dataset.load(dataset_file, DatasetUnpickler)
        if verbose:
            print(f"done in {c.elapsed_time():.2}s")
        return full_dataset
