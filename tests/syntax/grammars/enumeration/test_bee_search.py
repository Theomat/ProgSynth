from synth.syntax.grammars.enumeration.bee_search import (
    enumerate_prob_grammar,
)
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


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_beeSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    seen = set()
    for program in enumerate_prob_grammar(pcfg):
        assert program not in seen
        seen.add(program)
    assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
def test_order_beeSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    last = 1.0
    for program in enumerate_prob_grammar(pcfg):
        p = pcfg.probability(program)
        assert p <= last
        last = p


# @pytest.mark.parametrize("cfg", testdata)
# def test_merge(cfg: TTCFG) -> None:
#     pcfg = ProbDetGrammar.uniform(cfg)
#     seen = set()
#     for program in enumerate_prob_grammar(pcfg):
#         assert program not in seen
#         seen.add(program)
#     en = enumerate_prob_grammar(pcfg)
#     removed = dsl.parse_program("(+ 1 1)", auto_type("int"))
#     en.merge_program(dsl.parse_program("2", auto_type("int")), removed)
#     new_seen = set()
#     for program in en:
#         assert removed not in program
#         new_seen.add(program)
#     diff = seen.difference(new_seen)
#     for x in diff:
#         assert removed in x