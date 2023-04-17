from synth.syntax.grammars.ttcfg import TTCFG
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
max_sizes = [3, 7, 11]


@pytest.mark.parametrize("max_size", max_sizes)
def test_clean(max_size: int) -> None:
    cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    for rule in cfg.rules:
        for P in cfg.rules[rule]:
            if isinstance(P, Primitive):
                assert P.primitive != "non_reachable"
                assert P.primitive != "non_productive"
                assert P.primitive != "head"
            else:
                assert P.type == INT


@pytest.mark.parametrize("max_size,progs", [(1, 2), (3, 2 + 4), (5, 22)])
def test_size(max_size: int, progs: int) -> None:
    cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    size = cfg.programs()
    print(cfg)
    assert size == progs


@pytest.mark.parametrize("max_size", max_sizes)
def test_size_constraint(max_size: int) -> None:
    cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    size1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    res = size1
    while res.size() <= max_size:
        assert (
            res in cfg
        ), f"Program size:{res.size()} should be in the TTCFG max_size:{max_size}"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
    assert (
        res not in cfg
    ), f"Program size:{res.size()} should NOT be in the TTCFG max_size:{max_size}"


@pytest.mark.parametrize("max_occ", [3, 4, 5])
def test_at_most(max_occ: int) -> None:
    cfg = TTCFG.at_most_k(dsl, FunctionType(INT, INT), "+", max_occ)
    res = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    while res.depth() - 1 <= max_occ:
        assert (
            res in cfg
        ), f"Occurences:{res.depth() - 1} should be in the TTCFG max occurences:{max_occ}"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
    assert (
        res not in cfg
    ), f"Occurences:{res.depth() - 1} should NOT be in the TTCFG max occurences:{max_occ}"


@pytest.mark.parametrize("max_size", max_sizes)
def test_clean(max_size: int) -> None:
    cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    for rule in cfg.rules:
        for P in cfg.rules[rule]:
            if isinstance(P, Primitive):
                assert P.primitive != "non_reachable"
                assert P.primitive != "non_productive"
                assert P.primitive != "head"

    cpy = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    cpy.clean()
    assert cfg == cpy


def test_product() -> None:
    max_size = 3
    cfg1 = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size * 2)
    cfg2 = TTCFG.size_constraint(dsl, FunctionType(INT, INT), max_size)
    cfg = cfg1 * cfg2
    assert cfg
    size1 = dsl.parse_program("(+ 1 var0)", FunctionType(INT, INT))
    res = size1
    while res.size() <= max_size:
        assert (
            res in cfg
        ), f"Program size:{res.size()} should be in the TTCFG max_size:{max_size}"
        res = dsl.parse_program(f"(+ {res} var0)", FunctionType(INT, INT))
    assert (
        res not in cfg
    ), f"Program size:{res.size()} should NOT be in the TTCFG max_size:{max_size}"
