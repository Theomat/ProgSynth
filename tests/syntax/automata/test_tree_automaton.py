from itertools import product
from typing import Dict, Set, Tuple

from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
    Type,
)
from synth.syntax.grammars.grammar import DerivableProgram, NGram
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.grammars.cfg import CFG


def cfg2dfta(
    grammar: CFG,
) -> DFTA[Tuple[Type, int], DerivableProgram]:
    StateT = Tuple[Type, int]
    dfta_rules: Dict[Tuple[DerivableProgram, Tuple[StateT, ...]], StateT] = {}
    max_depth = grammar.max_program_depth()
    all_cases: Dict[
        Tuple[int, Tuple[Type, ...]], Set[Tuple[Tuple[Type, int], ...]]
    ] = {}
    for S in grammar.rules:
        for P in grammar.rules[S]:
            args = grammar.rules[S][P][0]
            if len(args) == 0:
                dfta_rules[(P, ())] = (P.type, 0)
            else:
                key = (len(args), tuple([arg[0] for arg in args]))
                if key not in all_cases:
                    all_cases[key] = set(
                        [
                            tuple(x)
                            for x in product(
                                *[
                                    [(arg[0], j) for j in range(max_depth)]
                                    for arg in args
                                ]
                            )
                        ]
                    )
                for nargs in all_cases[key]:
                    dfta_rules[(P, nargs)] = (
                        S[0],
                        max(i for _, i in nargs) + 1,
                    )
    r = grammar.type_request.returns()
    return DFTA(dfta_rules, {(r, x) for x in range(max_depth)})


import pytest

syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
max_depths = [3, 7, 11]


@pytest.mark.parametrize("max_depth", max_depths)
def test_reduce(max_depth: int) -> None:
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth, n_gram=1)
    dfta = cfg2dfta(cfg)
    dfta.reduce()
    for (P, args), dst in dfta.rules.items():
        assert not (
            all(x == 0 for x in args) and len(args) > 0
        ), f"Unreachable rule: {P} {args}"
        assert dst != max_depth, f"Unproductive rule: {P} {args} -> {dst}"


@pytest.mark.parametrize("max_depth", max_depths)
def test_states(max_depth: int) -> None:
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth, n_gram=1)
    dfta = cfg2dfta(cfg)
    dfta.reduce()
    for (P, args), dst in dfta.rules.items():
        if dst[1] < 1:
            continue
        state = (dst[0], ((NGram(1), dst[1] - 1), None))
        assert state in cfg.rules
        assert P in cfg.rules[state]
        assert all(a[0] == b[0] for a, b in zip(args, cfg.rules[state][P][0]))


@pytest.mark.parametrize("max_depth", max_depths)
def test_minimise(max_depth: int) -> None:
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    dfta = cfg2dfta(cfg)
    dfta.reduce()
    ndfta = dfta.minimise()
    for P, args in ndfta.rules:
        assert not (all(x == (0,) for x in args) and len(args) > 0)
