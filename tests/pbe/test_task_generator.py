from synth.generation.sampler import LexiconSampler
from synth.pbe.task_generator import TaskGenerator, basic_output_validator
from synth.semantic.evaluator import DSLEvaluator
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
validator = basic_output_validator({int: int_lexicon}, -1)


def test_gen() -> None:
    samples_lexicon = [2, 3, 4]
    pcfg = ProbDetGrammar.uniform(CFG.depth_constraint(dsl, type_req, max_depth))
    pcfg.init_sampling(20)
    g = TaskGenerator(
        LexiconSampler(int_lexicon, seed=10),
        DSLEvaluator(dsl.instantiate_semantics(semantics)),
        LexiconSampler([type_req], seed=10),
        LexiconSampler(samples_lexicon, [0.25, 0.5, 0.25], seed=10),
        {pcfg},
        validator,
    )
    for _ in range(100):
        task = g.generate_task()
        assert task.type_request == type_req
        assert len(task.specification.examples) in samples_lexicon
        assert task.solution
        assert task.solution.depth() <= max_depth
        for ex in task.specification.examples:
            assert validator(ex.output)
            assert len(ex.inputs) == 1
            assert all(x in int_lexicon for x in ex.inputs)


def test_seed() -> None:
    pcfg = ProbDetGrammar.uniform(CFG.depth_constraint(dsl, type_req, max_depth))
    pcfg.init_sampling(10)
    g1 = TaskGenerator(
        LexiconSampler(int_lexicon, seed=10),
        DSLEvaluator(dsl.instantiate_semantics(semantics)),
        LexiconSampler([type_req], seed=10),
        LexiconSampler([2, 3, 4], [0.25, 0.5, 0.25], seed=10),
        {pcfg},
        validator,
    )
    pcfg = ProbDetGrammar.uniform(CFG.depth_constraint(dsl, type_req, max_depth))
    pcfg.init_sampling(10)
    g2 = TaskGenerator(
        LexiconSampler(int_lexicon, seed=10),
        DSLEvaluator(dsl.instantiate_semantics(semantics)),
        LexiconSampler([type_req], seed=10),
        LexiconSampler([2, 3, 4], [0.25, 0.5, 0.25], seed=10),
        {pcfg},
        validator,
    )
    for _ in range(100):
        assert g1.generate_task() == g2.generate_task()


test_gen()
