from synth.semantic.evaluator import DSLEvaluator
from synth.specification import PBE, Example
from synth.syntax.grammars.enumeration.heap_search import enumerate_prob_grammar
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType
from synth.pbe.solvers import NaivePBESolver, ObsEqPBESolver, CutoffPBESolver, PBESolver

import pytest

from synth.task import Task


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}

semantics = {"+": lambda x: lambda y: x + y, "-": lambda x: lambda y: x - y, "1": 1}


type_req = FunctionType(INT, INT)
int_lexicon = list(range(-100, 100))
max_depth = 4
dsl = DSL(syntax)
evaluator = DSLEvaluator(dsl.instantiate_semantics(semantics))
testdata = [
    NaivePBESolver(evaluator),
    ObsEqPBESolver(evaluator),
    CutoffPBESolver(evaluator),
]

cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), 4)
pcfg = ProbDetGrammar.uniform(cfg)


tasks = [
    Task(cfg.type_request, PBE([Example([x], x + 2) for x in [3, 4, 9, 12]])),
    Task(cfg.type_request, PBE([Example([x], x - 2) for x in [3, 4, 9, 12]])),
]


@pytest.mark.parametrize("solver", testdata)
def test_solving(solver: PBESolver) -> None:
    for task in tasks:
        failed = True
        for program in solver.solve(task, enumerate_prob_grammar(pcfg), 10):
            for example in task.specification.examples:
                assert evaluator.eval(program, example.inputs) == example.output
            failed = False
            assert solver._score > 0
            break
        assert not failed
