from synth.syntax.grammars.ttcfg import TTCFG
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


def test_from_dsl() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"
                    assert P.primitive != "head"
                else:
                    assert P.type == INT


def test_depth_constraint() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
        depth1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
        res = depth1
        while res.depth() <= max_depth:
            assert (
                res in cfg
            ), f"Program depth:{res.depth()} should be in the TTCFG max_depth:{max_depth}"
            res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
        assert (
            res not in cfg
        ), f"Program depth:{res.depth()} should NOT be in the TTCFG max_depth:{max_depth}"


def test_size_constraint() -> None:
    dsl = DSL(syntax)
    for max_size in [3, 7, 11]:
        cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
        depth1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
        res = depth1
        while res.length() <= max_size:
            assert (
                res in cfg
            ), f"Program size:{res.length()} should be in the TTCFG max_size:{max_size}"
            res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
        assert (
            res not in cfg
        ), f"Program size:{res.length()} should NOT be in the TTCFG max_size:{max_size}"


def test_at_most() -> None:
    dsl = DSL(syntax)
    for max_occ in [3, 7, 11]:
        cfg = TTCFG.at_most_k(dsl, FunctionType(INT, INT), "+", max_occ)
        depth1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
        res = depth1
        while res.depth() - 1 <= max_occ:
            assert (
                res in cfg
            ), f"Occurences:{res.depth() - 1} should be in the TTCFG max occurences:{max_occ}"
            res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
        assert (
            res not in cfg
        ), f"Occurences:{res.depth() - 1} should NOT be in the TTCFG max occurences:{max_occ}"


def test_clean() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"

        cpy = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
        cpy.clean()
        assert cfg == cpy


def test_product() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg1 = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth * 2)
    cfg2 = TTCFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)
    cfg = cfg1 * cfg2
    assert cfg
    depth1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    res = depth1
    while res.depth() <= max_depth:
        assert (
            res in cfg
        ), f"Program depth:{res.depth()} should be in the TTCFG max_depth:{max_depth}"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
    assert (
        res not in cfg
    ), f"Program depth:{res.depth()} should NOT be in the TTCFG max_depth:{max_depth}"
