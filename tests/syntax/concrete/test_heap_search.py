from synth.syntax.concrete.heap_search import enumerate_pcfg
from synth.syntax.concrete.concrete_cfg import ConcreteCFG
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.syntax.dsl import DSL
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


def test_unicity() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    seen = set()
    for program in enumerate_pcfg(pcfg):
        assert program not in seen
        seen.add(program)


def test_order() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    last = 1.0
    for program in enumerate_pcfg(pcfg):
        p = pcfg.probability(program)
        assert p <= last
        last = p
