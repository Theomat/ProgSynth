from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive
from synth.syntax.type_system import (
    INT,
    STRING,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
)

import pytest


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
dsl.instantiate_polymorphic_types()
max_depths = [3, 7, 11]


@pytest.mark.parametrize("max_depth", max_depths)
def test_size(max_depth: int) -> None:
    ucfg = UCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    assert cfg.programs() == ucfg.programs()


@pytest.mark.parametrize("max_depth", max_depths)
def test_clean(max_depth: int) -> None:
    cfg = UCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    for rule in cfg.rules:
        assert rule[1][1] <= max_depth
        for P in cfg.rules[rule]:
            if isinstance(P, Primitive):
                assert P.primitive != "non_reachable"
                assert P.primitive != "non_productive"
                assert P.primitive != "head"


@pytest.mark.parametrize("max_depth", max_depths)
def test_depth_constraint(max_depth: int) -> None:
    cfg = UCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
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
