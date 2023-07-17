from typing import TypeVar
from synth.pruning.constraints import add_dfta_constraints
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.enumeration.heap_search import enumerate_prob_grammar
from synth.syntax.grammars.enumeration.u_heap_search import (
    Bucket,
    enumerate_prob_u_grammar,
    enumerate_bucket_prob_u_grammar,
)
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
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
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "2": INT,
    "0": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
dsl.instantiate_polymorphic_types()
testdata = [
    UCFG.depth_constraint(dsl, FunctionType(INT, INT), 3),
    UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(
            CFG.depth_constraint(dsl, FunctionType(INT, INT), 3),
            ["(+ 1 ^0)", "(- _ ^0)"],
        ),
        2,
    ),
]


def test_equality() -> None:
    base = CFG.depth_constraint(dsl, FunctionType(INT, INT), 3, min_variable_depth=0)
    ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(base, [], progress=False),
        2,
    )

    seen = set()
    for program in enumerate_prob_u_grammar(ProbUGrammar.uniform(ucfg)):
        seen.add(program)
    seen2 = set()
    for program in enumerate_prob_grammar(ProbDetGrammar.uniform(base)):
        seen2.add(program)
    assert len(seen.difference(seen2)) == 0


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_heapSearch(cfg: UCFG) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    seen = set()
    for program in enumerate_prob_u_grammar(pcfg):
        assert program not in seen
        seen.add(program)
    assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
def test_order_heapSearch(cfg: UCFG) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    last = 1.0
    for program in enumerate_prob_u_grammar(pcfg):
        p = pcfg.probability(program)
        assert (p - last) <= 1e-7
        last = p


@pytest.mark.parametrize("cfg", testdata)
def test_threshold(cfg: UCFG) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    threshold = 0.15
    seen = set()
    for program in enumerate_prob_u_grammar(pcfg):
        p = pcfg.probability(program)
        if p <= threshold:
            break
        seen.add(p)
    seent = set()
    for program in enumerate_prob_u_grammar(pcfg, threshold):
        p = pcfg.probability(program)
        assert p > threshold
        seent.add(p)

    assert len(seent.symmetric_difference(seen)) == 0


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_bucketSearch(cfg: UCFG) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    for bucketSize in range(3, 10):
        seen = set()
        for program in enumerate_bucket_prob_u_grammar(pcfg, bucket_size=bucketSize):
            assert program not in seen
            seen.add(program)
        assert len(seen) == cfg.programs()


U = TypeVar("U")


@pytest.mark.parametrize("cfg", testdata)
def test_order_bucketSearch(cfg: UCFG[U]) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    for bucketSize in range(3, 10):
        last = Bucket(bucketSize)
        for program in enumerate_bucket_prob_u_grammar(pcfg, bucket_size=bucketSize):
            outs = pcfg.reduce_derivations(
                lambda b, S, P, v: b.add_prob_uniform(
                    pcfg.probabilities[S][P][tuple(v)]
                ),
                Bucket(bucketSize),
                program,
            )
            assert len(outs) == 1
            p = outs[0]
            assert p.size == bucketSize
            assert p >= last or last == Bucket(bucketSize)
            last = p


@pytest.mark.parametrize("cfg", testdata)
def test_merge(cfg: UCFG[U]) -> None:
    pcfg = ProbUGrammar.uniform(cfg)
    seen = set()
    for program in enumerate_prob_u_grammar(pcfg):
        assert program not in seen
        seen.add(program)
    en = enumerate_prob_u_grammar(pcfg)
    removed = dsl.parse_program("(- 1 1)", auto_type("int"))
    en.merge_program(dsl.parse_program("0", auto_type("int")), removed)
    new_seen = set()
    print(cfg)
    for program in en:
        assert removed not in program
        new_seen.add(program)
    diff = seen.difference(new_seen)
    for x in diff:
        assert removed in x
