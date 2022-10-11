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

user = [
    "(+ _ ^*)",
    "(* _ ^+)",
    "(if ^not _ _)",
    "(le? ^- _)",
    "(gt? ^- _)",
]

equations = [
    "(+ ^+,0 ^0)",
    "(- _ ^0)",
    "(* ^*,1,0 ^2,1,0)",
    "(max ^max _)",
    "(min ^min _)",
    "(is-mod ^0,1 _)",
    "(mod ^0,1 _)",
    "(length ^range,cdr,map,cons)",
    "(not ^not)",
    "(index ^0 _)",
    "(empty? ^range,map)",
    "(range ^0)",
    "(car ^range)",
    "(cdr ^cons,map,filter)",
]

seed = 1
# ===============================================================
# Fill here with your grammars
# ===============================================================


def produce_grammars(depth: int) -> Dict[str, int]:
    cfg = CFG.depth_constraint(dsl, type_request, depth)
    if depth == 3:

        ttcfg = add_constraints(cfg, user + equations, progress=False)
    ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, dsl_module.constraints, progress=True), 2
    )
    uucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, user, progress=True), 2
    )
    eucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, equations, progress=True), 2
    )
    return {
        "cfg": cfg.programs(),
        "ucfg": ucfg.programs(),
        "user-ucfg": uucfg.programs(),
        "equations-ucfg": eucfg.programs(),
        "ttcfg": ttcfg.programs_stochastic(cfg, 100000, seed) * cfg.programs()
        if depth == 3
        else -1,
    }


def int2scientific(i: int) -> str:
    try:
        return f"{i:.2e}"
    except OverflowError:
        s = int2scientific(i // int(10**100))
        e_index = s.index("e") + 1
        exp = int(s[e_index + 1 :]) + 100
        return s[:e_index] + "+" + str(exp)


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
    output.append([depth] + [f"{int2scientific(all_grammars[name])}" for name in order])

file = f"./{dsl_name}_grammar_sizes_{min_depth}_to_{max_depth}.csv"
with open(file, "w") as fd:
    csv.writer(fd).writerows(output)
print("saved to", file)
