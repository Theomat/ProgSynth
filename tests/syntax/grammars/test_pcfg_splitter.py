from synth.syntax.grammars.heap_search import enumerate_pcfg
from synth.syntax.grammars.pcfg_splitter import split
from synth.syntax.grammars.concrete_cfg import ConcreteCFG
from synth.syntax.grammars.concrete_pcfg import ConcretePCFG
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
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}


def test_unicity() -> None:
    dsl = DSL(syntax)
    max_depth = 4
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    for splits in [2, 4, 5]:
        fragments, _ = split(pcfg, splits, desired_ratio=1.05)
        seen = set()
        for sub_pcfg in fragments:
            for program in enumerate_pcfg(sub_pcfg):
                assert program not in seen
                seen.add(program)


def prout_test_none_missing() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    seen = set()
    for program in enumerate_pcfg(pcfg):
        seen.add(program)
    for splits in [2, 4, 5]:
        fragments, _ = split(pcfg, splits, desired_ratio=1.05)
        new_seen = set()
        for sub_pcfg in fragments:
            a = set()
            for program in enumerate_pcfg(sub_pcfg):
                a.add(program)
            new_seen |= a
        assert len(seen.difference(new_seen)) == 0, seen.difference(new_seen)
        assert len(new_seen.difference(seen)) == 0, new_seen.difference(seen)
