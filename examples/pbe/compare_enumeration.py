from typing import Callable, Iterable, List, Tuple, Union
import csv
import time
import argparse

from dsl_loader import add_dsl_choice_arg, load_DSL

from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    CFG,
    DSL,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    ProgramEnumerator,
    Type,
    auto_type,
)

import tqdm

SEARCH_ALGOS = {
    "beap_search": (bps_enumerate_prob_grammar, None),
    "heap_search": (hs_enumerate_prob_grammar, hs_enumerate_prob_u_grammar),
    "bee_search": (bs_enumerate_prob_grammar, None),
}

parser = argparse.ArgumentParser(
    description="Compare search algorithms", fromfile_prefix_chars="@"
)
add_dsl_choice_arg(parser)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="./enumeration.csv",
    help="output file (default: './enumeration.csv')",
)
parser.add_argument(
    dest="n", metavar="programs", type=int, help="number of programs to be enumerated"
)
parser.add_argument("type", type=str, help="type request")
parser.add_argument(
    "-d", "--depth", type=int, default=5, help="max program depth (default: 5)"
)
parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
parser.add_argument("--scaling", action="store_true", help="scaling with non terminals")


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
output_file: str = parameters.output
str_tr: str = parameters.type
programs: int = parameters.n
max_depth: int = parameters.depth
seed: int = parameters.seed
scaling: bool = parameters.scaling

file_name = output_file[: -len(".csv")]
suffix = (
    f"_dsl_{dsl_name}_seed_{seed}_depth_{max_depth}"
    if not scaling
    else f"_dsl_scaling_seed_{seed}_depth_{max_depth}"
)
if not file_name.endswith(suffix):
    file_name += suffix
output_file = file_name + ".csv"


# ================================
# Load constants specific to dataset
# ================================


def save(trace: Iterable) -> None:
    with open(output_file, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(trace)


# Enumeration methods =====================================================
def enumerative_search(
    pcfg: ProbDetGrammar,
    name: str,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    programs: int,
    datum_each: int = 50000,
    timeout: int = 300,
    average_only: bool = False,
) -> List[Tuple[str, int, float, int, int, int]]:
    out = []
    n = 0
    non_terminals = len(pcfg.rules)
    pbar = tqdm.tqdm(total=programs, desc=name)
    start = time.perf_counter_ns()
    enumerator = custom_enumerate(pcfg)
    for program in enumerator.generator():
        n += 1
        if n % datum_each == 0 or n >= programs:
            used_time = time.perf_counter_ns() - start
            bef = time.perf_counter_ns()
            if used_time >= timeout * 1e9:
                break
            out.append(
                (
                    name,
                    non_terminals,
                    used_time / 1e9,
                    n,
                    enumerator.programs_in_queues(),
                    enumerator.programs_in_banks(),
                )
            )
            pbar.update(datum_each)
            if n >= programs:
                pbar.close()
                break
            start -= time.perf_counter_ns() - bef
    if n < programs:
        pbar.close()
    if average_only and len(out) > 0:
        progs = out[-1][3]
        factor = datum_each / progs
        new_out = [
            (
                name,
                non_terminals,
                out[-1][2] * factor,
                datum_each,
                out[-1][-2] * factor,
                out[-1][-1] * factor,
            )
        ]
        out = new_out
    return out


# Main ====================================================================

if __name__ == "__main__":
    import sys

    trace = [("search", "non_terminals", "time", "programs", "queue", "bank")]

    if not scaling:
        dsl_module = load_DSL(dsl_name)
        dsl: DSL = dsl_module.dsl
        constant_types = getattr(dsl, "constant_types", set())

        n_gram = 2 if max_depth > 0 else 1
        try:
            cfg = CFG.depth_constraint(
                dsl,
                auto_type(str_tr),
                max_depth,
                n_gram=n_gram,
                constant_types=constant_types,
            )
            pcfg = ProbDetGrammar.random(cfg, seed=seed)
        except KeyError:
            print(
                f"failed to instantiate a non empty grammar for dsl {dsl} and type: {str_tr}",
                file=sys.stderr,
            )
            sys.exit(1)
        for name, (enum, _) in SEARCH_ALGOS.items():
            trace += enumerative_search(pcfg, name, enum, programs)
    else:
        dsl = DSL(
            auto_type(
                {
                    "+": "int -> int -> int",
                    "-": "int -> int -> int",
                    "*": "int -> int -> int",
                    "1": "int",
                }
            )
        )
        for ngram in [1, 2]:
            for depth in range(1, max_depth + 1):
                try:
                    cfg = CFG.depth_constraint(
                        dsl,
                        auto_type("int -> int"),
                        depth,
                        n_gram=ngram,
                    )
                    print("depth:", depth, "non terminals:", len(cfg.rules))
                    pcfg = ProbDetGrammar.random(cfg, seed=seed)
                except KeyError:
                    print(
                        f"failed to instantiate a non empty grammar for dsl {dsl} and type: {str_tr}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                for name, (enum, _) in SEARCH_ALGOS.items():
                    trace += enumerative_search(
                        pcfg, name, enum, programs, average_only=True
                    )
    save(trace)
    print("csv file was saved as:", output_file)
