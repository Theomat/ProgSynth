from synth.syntax.grammars.enumeration.constant_delay import (
    enumerate_prob_grammar,
)
from synth.syntax.grammars.enumeration.beap_search import enumerate_prob_grammar as enumerate

from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType, auto_type

import numpy as np

import pytest


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "2": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
dsl.instantiate_polymorphic_types()
testdata = [
    CFG.depth_constraint(dsl, FunctionType(INT, INT), 3),
    CFG.depth_constraint(dsl, FunctionType(INT, INT), 4),
]
kvals = [4, 16, 64]
precision = [1e-3, 1e-5, 1e-8]

@pytest.mark.parametrize("cfg", testdata)
@pytest.mark.parametrize("k", kvals)
@pytest.mark.parametrize("precis", precision)
def test_equality_beep_search(cfg: TTCFG, k : int, precis: float) -> None:
    pcfg = ProbDetGrammar.random(cfg, seed=1)
    g1 = enumerate(pcfg)
    g2 = enumerate_prob_grammar(pcfg, k, precis)
    for p1, p2 in zip(g1, g2):
        assert abs(pcfg.probability(p1) - pcfg.probability(p2)) <= 1e-8 or abs(pcfg.probability(p1) / pcfg.probability(p2)) <= 1 + precis

@pytest.mark.parametrize("cfg", testdata)
@pytest.mark.parametrize("k", kvals)
@pytest.mark.parametrize("precis", precision)
def test_unicity_beep_search(cfg: TTCFG, k : int, precis: float) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    seen = set()
    for program in enumerate_prob_grammar(pcfg, k, precis):
        assert program not in seen
        seen.add(program)
    # print(pcfg.grammar)
    assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
@pytest.mark.parametrize("k", kvals)
@pytest.mark.parametrize("precis", precision)
def test_order_beep_search(cfg: TTCFG, k : int, precis: float) -> None:
    pcfg = ProbDetGrammar.random(cfg, seed=4)
    last = 1.0
    for program in enumerate_prob_grammar(pcfg, k, precis):
        p = pcfg.probability(program)
        assert p <= last or abs(p / last) <= 1 + precis
        last = p


@pytest.mark.parametrize("k", kvals)
@pytest.mark.parametrize("precis", precision)
def test_infinite(k : int, precis: float) -> None:
    pcfg = ProbDetGrammar.random(
        CFG.infinite(dsl, testdata[0].type_request, n_gram=1), 1
    )
    count = 10000
    last = 1.0
    for program in enumerate_prob_grammar(pcfg, k, precis):
        count -= 1
        p = pcfg.probability(program)
        assert -1e-12 <= last - p or abs(p / last) <= 1 + precis, f"failed at program n°{count}:{program}, p={p} last={last}, p={np.log(p)} last={np.log(last)}"
        last = p
        if count < 0:
            break
    assert count == -1
