from typing import Dict, List as TList
import csv
import tqdm

from examples.pbe.dsl_loader import load_DSL

from synth.syntax import (
    CFG,
    DSL,
)
from synth.syntax.grammars import Grammar, UCFG
from synth.syntax.type_system import INT, FunctionType, Type, List
from synth.pruning.constraints import add_constraints, add_dfta_constraints

# ===============================================================
# Change parameters here
# ===============================================================
min_depth: int = 3
max_depth: int = 7
type_request: Type = FunctionType(List(INT), List(INT))
dsl_name: str = "dreamcoder"
dsl_module = load_DSL(dsl_name)
dsl: DSL = dsl_module.dsl
dsl.instantiate_polymorphic_types()
constraints: TList[str] = dsl_module.constraints
# ===============================================================
# Fill here with your grammars
# ===============================================================


def produce_grammars(depth: int) -> Dict[str, Grammar]:
    cfg = CFG.depth_constraint(dsl, type_request, depth)
    ttcfg = add_constraints(cfg, constraints, progress=False)
    ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, constraints, progress=False), 2
    )
    return {"cfg": cfg, "ucfg": ucfg, "ttcfg": ttcfg}


# ===============================================================
# No changes under here
# ===============================================================
output = []
order = []
for depth in tqdm.trange(min_depth, max_depth + 1):

    all_grammars = produce_grammars(depth)
    if len(output) == 0:
        order = list(all_grammars.keys())
        output.append(["depth"] + order)
    output.append([depth] + [f"{all_grammars[name].programs():e}" for name in order])

file = f"./grammar_sizes_{min_depth}_to_{max_depth}.csv"
with open(file, "w") as fd:
    csv.writer(fd).writerows(output)
print("saved to", file)
