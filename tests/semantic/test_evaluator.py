from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.semantic.evaluator import DSLEvaluator, __tuplify__
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
    "+1": FunctionType(INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "non_productive": FunctionType(INT, STRING),
}

semantics = {
    "+1": lambda x: x + 1,
}
max_depth = 4
dsl = DSL(syntax)
cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), max_depth)


def test_eval() -> None:
    eval = DSLEvaluator(dsl.instantiate_semantics(semantics))
    pcfg = ProbDetGrammar.uniform(cfg)
    pcfg.init_sampling(0)
    for _ in range(100):
        program = pcfg.sample_program()
        try:
            for i in range(-25, 25):
                assert eval.eval(program, [i]) == program.size() + i - 1
        except Exception as e:
            assert False, e


def test_supports_list() -> None:
    eval = DSLEvaluator(dsl.instantiate_semantics(semantics))
    pcfg = ProbDetGrammar.uniform(cfg)
    pcfg.init_sampling(0)
    for _ in range(100):
        program = pcfg.sample_program()
        try:
            for i in range(-25, 25):
                assert eval.eval(program, [i, [i]]) == program.size() + i - 1
        except Exception as e:
            assert False, e


def test_use_cache() -> None:
    eval = DSLEvaluator(dsl.instantiate_semantics(semantics))
    pcfg = ProbDetGrammar.uniform(cfg)
    pcfg.init_sampling(0)
    for _ in range(100):
        program = pcfg.sample_program()
        try:
            for i in range(-25, 25):
                assert eval.eval(program, [i]) == program.size() + i - 1
                assert eval._cache[__tuplify__([i])][program] == program.size() + i - 1
        except Exception as e:
            assert False, e
