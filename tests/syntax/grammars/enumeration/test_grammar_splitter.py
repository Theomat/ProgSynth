import pytest
from synth.syntax.grammars.enumeration.u_heap_search import enumerate_prob_u_grammar
from synth.syntax.grammars.enumeration.grammar_splitter import split
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
pucfg = ProbUGrammar.uniform(UCFG.depth_constraint(dsl, FunctionType(INT, INT), 3))
testdata = list(range(2, 5))
seen = set()
for program in enumerate_prob_u_grammar(pucfg):
    seen.add(program)


@pytest.mark.parametrize("splits", testdata)
def test_unicity(splits: int) -> None:
    fragments, _ = split(pucfg, splits, desired_ratio=1.05)
    seen = set()
    for sub_pcfg in fragments:
        print(sub_pcfg)
        for program in enumerate_prob_u_grammar(sub_pcfg):
            assert program not in seen
            seen.add(program)


@pytest.mark.parametrize("splits", testdata)
def test_none_missing(splits: int) -> None:
    fragments, _ = split(pucfg, splits, desired_ratio=1.05)
    new_seen = set()
    for sub_pcfg in fragments:
        a = set()
        for program in enumerate_prob_u_grammar(sub_pcfg):
            a.add(program)
        new_seen |= a
    assert len(new_seen.difference(seen)) == 0, new_seen.difference(seen)
    assert len(seen.difference(new_seen)) == 0, seen.difference(new_seen)
