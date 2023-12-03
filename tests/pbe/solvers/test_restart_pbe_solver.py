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
from synth.pbe.solvers.restart_pbe_solver import RestartPBESolver

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
max_depth = 4
dsl = DSL(syntax)
evaluator = DSLEvaluator(dsl.instantiate_semantics(semantics))
testdata = [
    NaivePBESolver(evaluator),
    ObsEqPBESolver(evaluator),
    CutoffPBESolver(evaluator),
]

cfg = CFG.depth_constraint(dsl, type_req, max_depth)
pcfg = ProbDetGrammar.uniform(cfg)


tasks = [
    Task(cfg.type_request, PBE([Example([x], x + 2) for x in range(50)])),
    Task(cfg.type_request, PBE([Example([x], x - 2) for x in range(50)])),
]


@pytest.mark.parametrize("solver", testdata)
def test_solving(solver: PBESolver) -> None:
    real_solver = RestartPBESolver(
        solver.evaluator,
        lambda *args, **kwargs: solver,
        restart_criterion=lambda self: len(self._data) - self._last_size > 3,
    )
    for task in tasks:
        failed = True
        real_solver.reset_stats()
        for program in real_solver.solve(task, enumerate_prob_grammar(pcfg), 5):
            for example in task.specification.examples:
                assert evaluator.eval(program, example.inputs) == example.output
            failed = False
            if real_solver._restarts <= 0:
                continue
            break
        assert not failed
