from typing import Dict
import csv
import tqdm
from experiments.towers.towers_base import (
    BLOCK,
    syntax,
    user,
    equations,
    sketch,
    dfta_constraints,
)

from synth.syntax import (
    CFG,
    DSL,
)
from synth.syntax.grammars import UCFG
from synth.syntax.type_system import INT, FunctionType, Type
from synth.pruning.constraints import add_constraints, add_dfta_constraints

# ===============================================================
# Change parameters here
# ===============================================================
min_depth: int = 3
max_depth: int = 7
type_request: Type = FunctionType(INT, INT, BLOCK)
dsl_name: str = "towers"
dsl = DSL(syntax)
seed = 1
# ===============================================================
# Fill here with your grammars
# ===============================================================


def produce_grammars(depth: int) -> Dict[str, int]:
    cfg = CFG.depth_constraint(dsl, type_request, depth)
    if depth == 3:
        ttcfg = add_constraints(
            cfg,
            user + equations,
            sketch,
            progress=False,
        )
    ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(
            cfg, user + equations + dfta_constraints, sketch, progress=True
        ),
        2,
    )
    uucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, user + dfta_constraints, sketch, progress=True),
        2,
    )
    eucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, equations, progress=True),
        2,
    )
    return {
        "cfg": cfg.programs(),
        "user-ucfg": uucfg.programs(),
        "equations-ucfg": eucfg.programs(),
        "ucfg": ucfg.programs(),
        "ttcfg": ttcfg.programs_stochastic(cfg, 100000, seed) * cfg.programs()
        if depth == 3
        else -1,
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
    output.append([depth] + [f"{all_grammars[name]:.2e}" for name in order])

file = f"./{dsl_name}_grammar_sizes_{min_depth}_to_{max_depth}.csv"
with open(file, "w") as fd:
    csv.writer(fd).writerows(output)
print("saved to", file)
