from typing import Callable, Iterable, List, Tuple, Union
import csv
import time

from dsl_loader import add_dsl_choice_arg, load_DSL


from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    CFG,
    DSL,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bpluss_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    ProgramEnumerator,
    Type,
)

import tqdm

import argparse

from synth.syntax.type_helper import auto_type


SEARCH_ALGOS = {
    "beep_search": (bps_enumerate_prob_grammar, None),
    "heap_search": (hs_enumerate_prob_grammar, hs_enumerate_prob_u_grammar),
    # "bee_search": (bs_enumerate_prob_grammar, None),
    "bee+_search": (bpluss_enumerate_prob_grammar, None),
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


parameters = parser.parse_args()
dsl_name: str = parameters.dsl
output_file: str = parameters.output
str_tr: str = parameters.type
programs: int = parameters.n
max_depth: int = parameters.depth
seed: int = parameters.seed

file_name = output_file[: -len(".csv")]
suffix = f"_dsl_{dsl_name}_seed_{seed}_depth_{max_depth}"
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
    cfgs: List[CFG],
    name: str,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    programs: int,
    datum_each: int = 1000,
) -> List[Tuple[str, Type, float, int, int, int]]:
    out = []
    for cfg in cfgs:
        n = 0
        pbar = tqdm.tqdm(total=programs, desc=name)
        start = time.perf_counter()
        enumerator = custom_enumerate(cfg)
        for program in enumerator.generator():
            n += 1
            if n % datum_each == 0 or n >= programs:
                used_time = time.perf_counter() - start
                bef = time.perf_counter()
                out.append(
                    (
                        name,
                        cfg.type_request,
                        used_time,
                        n,
                        enumerator.programs_in_queues(),
                        enumerator.programs_in_banks(),
                    )
                )
                pbar.update(datum_each)
                if n >= programs:
                    pbar.close()
                    break
                start -= time.perf_counter() - bef
    return out


# Main ====================================================================

if __name__ == "__main__":
    dsl_module = load_DSL(dsl_name)
    dsl: DSL = dsl_module.dsl

    base_types = set()
    for p in dsl.list_primitives:
        base_types |= p.type.decompose_type()[0]

    type_requests = [auto_type(str_tr)]
    pcfgs = []
    n_gram = 2 if max_depth > 0 else 1
    for tr in type_requests:
        try:
            cfg = CFG.depth_constraint(dsl, tr, max_depth, n_gram=n_gram)
            pcfg = ProbDetGrammar.random(cfg, seed=seed)
            pcfgs.append(pcfg)
        except KeyError:
            pass
    assert len(pcfgs) > 0, f"failed to generate any simple CFGs for {type_requests}"
    print(f"generated {len(pcfgs)} CFGs")
    trace = [("search", "type", "time", "programs", "queue", "bank")]
    for name, (enum, _) in SEARCH_ALGOS.items():
        trace += enumerative_search(pcfgs, name, enum, programs)
    save(trace)
    print("csv file was saved as:", output_file)
