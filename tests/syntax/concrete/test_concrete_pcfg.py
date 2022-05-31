import numpy as np

from synth.syntax.concrete.concrete_cfg import ConcreteCFG
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
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


def test_from_cfg() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        pcfg = ConcretePCFG.uniform(cfg)
        for rule in pcfg.rules:
            n = len(pcfg.rules[rule])
            for P in pcfg.rules[rule]:
                _, prob = pcfg.rules[rule][P]
                assert np.isclose(prob, 1 / n)

        cpy = ConcretePCFG.uniform(cfg)
        assert cpy == pcfg


def test_clean() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        pcfg = ConcretePCFG.uniform(cfg)
        for rule in pcfg.rules:
            assert rule.depth <= max_depth
            for P in pcfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"

        cpy = ConcretePCFG.uniform(cfg)
        cpy.clean()
        assert pcfg == cpy


def test_ready_for_sampling() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        pcfg = ConcretePCFG.uniform(cfg)
        assert not pcfg.ready_for_sampling
        pcfg.init_sampling()
        assert pcfg.ready_for_sampling


def test_seeding() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        seed = 100
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        pcfg = ConcretePCFG.uniform(cfg)
        pcfg.init_sampling(seed)
        g1 = pcfg.sampling()
        cpy = ConcretePCFG.uniform(cfg)
        cpy.init_sampling(seed)
        assert pcfg == cpy
        g2 = cpy.sampling()
        for _ in range(1000):
            p1, p2 = next(g1), next(g2)
            assert p1 == p2, f"[n°{_}]: {p1} != {p2}"


def test_depth() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        pcfg = ConcretePCFG.uniform(cfg)
        pcfg.init_sampling(0)
        g = pcfg.sampling()
        for _ in range(1000):
            assert next(g).depth() <= max_depth
