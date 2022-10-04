from typing import Dict
import csv
import tqdm
from experiments.towers.towers_base import (
    BLOCK,
    syntax,
    constraints,
    sketch,
    dfta_constraints,
)

from synth.syntax import (
    CFG,
    DSL,
)
from synth.syntax.grammars import Grammar, UCFG
from synth.syntax.type_system import INT, FunctionType, Type
from synth.pruning.constraints import add_constraints, add_dfta_constraints

# ===============================================================
# Change parameters here
# ===============================================================
min_depth: int = 3
max_depth: int = 4
type_request: Type = FunctionType(INT, INT, BLOCK)
dsl_name: str = "towers"
dsl = DSL(syntax)
seed = 1
# ===============================================================
# Fill here with your grammars
# ===============================================================


def produce_grammars(depth: int) -> Dict[str, int]:
    cfg = CFG.depth_constraint(dsl, type_request, depth)
    ttcfg = add_constraints(
        cfg,
        constraints,
        sketch,
        progress=True,
    )
    ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(
            cfg, constraints + dfta_constraints, sketch, progress=True
        ),
        2,
    )
    return {
        "cfg": cfg.programs(),
        "ucfg": ucfg.programs(),
        "ttcfg": ttcfg.programs_stochastic(cfg, 100000, seed) * cfg.programs(),
    }


# ===============================================================
# No changes under here
# ===============================================================
output = []
order = []
for depth in tqdm.trange(min_depth, max_depth + 1, desc="depth"):

    all_grammars = produce_grammars(depth)
    if len(output) == 0:
        order = list(all_grammars.keys())
        output.append(["depth"] + order)
    output.append([depth] + [f"{all_grammars[name]:e}" for name in order])

file = f"./{dsl_name}_grammar_sizes_{min_depth}_to_{max_depth}.csv"
with open(file, "w") as fd:
    csv.writer(fd).writerows(output)
print("saved to", file)
