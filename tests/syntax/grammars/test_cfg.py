from synth.syntax.grammars.cfg import CFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive
from synth.syntax.type_system import (
    INT,
    STRING,
    Arrow,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType

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


def test_function_as_variable() -> None:
    dsl = DSL(syntax)
    max_depth = 5
    cfg = CFG.depth_constraint(dsl, FunctionType(Arrow(INT, INT), INT), max_depth)
    assert cfg.programs() > 0


@pytest.mark.parametrize("max_depth", max_depths)
def test_clean(max_depth: int) -> None:
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    for rule in cfg.rules:
        assert rule[1][0][1] <= max_depth
        for P in cfg.rules[rule]:
            if isinstance(P, Primitive):
                assert P.primitive != "non_reachable"
                assert P.primitive != "non_productive"
                assert P.primitive != "head"


@pytest.mark.parametrize("max_depth", max_depths)
def test_depth_constraint(max_depth: int) -> None:
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    res = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    print(cfg)
    while res.depth() <= max_depth:
        assert (
            res in cfg
        ), f"Program depth:{res.depth()} should be in the TTCFG max_depth:{max_depth}"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
    assert (
        res not in cfg
    ), f"Program depth:{res.depth()} should NOT be in the TTCFG max_depth:{max_depth}"


def test_infinite() -> None:
    cfg = CFG.infinite(dsl, FunctionType(INT, INT), n_gram=2)
    res = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    print(cfg)
    while res.depth() <= 30:
        assert (
            res in cfg
        ), f"Program depth:{res.depth()} should be in the infinite TTCFG"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
