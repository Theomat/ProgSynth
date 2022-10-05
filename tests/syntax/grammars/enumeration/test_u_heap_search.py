from typing import TypeVar
from synth.pruning.constraints import add_dfta_constraints
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.enumeration.u_heap_search import (
    Bucket,
    enumerate_prob_u_grammar,
    enumerate_bucket_prob_u_grammar,
)
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.dsl import DSL
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
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
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
        1,
    ),
]


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
