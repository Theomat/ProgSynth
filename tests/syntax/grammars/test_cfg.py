from synth.syntax.grammars.cfg import CFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive
from synth.syntax.type_system import (
    INT,
    STRING,
    Arrow,
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


def test_function_as_variable() -> None:
    dsl = DSL(syntax)
    max_depth = 5
    cfg = CFG.depth_constraint(dsl, FunctionType(Arrow(INT, INT), INT), max_depth)
    assert cfg.programs() > 0


def test_clean() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            assert rule[1][0][1] <= max_depth
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"
                    assert P.primitive != "head"


def test_depth_constraint() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
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
