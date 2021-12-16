from synth.syntax.concrete.concrete_cfg import ConcreteCFG
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


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}


def test_from_dsl() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            assert rule.depth <= max_depth
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                else:
                    assert P.type == INT


def test_clean() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            assert rule.depth <= max_depth
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"

        cpy = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        cpy.clean()
        assert cfg == cpy
