from synth.syntax.concrete.heap_search import (
    Bucket,
    enumerate_pcfg,
    enumerate_bucket_pcfg,
)
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
    # "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    # "1": INT,
    # "2": INT,
    "non_productive": FunctionType(INT, STRING),
}


def test_unicity_heapSearch() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    seen = set()
    for program in enumerate_pcfg(pcfg):
        assert program not in seen
        seen.add(program)
    assert len(seen) == cfg.size()


def test_order_heapSearch() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    last = 1.0
    for program in enumerate_pcfg(pcfg):
        p = pcfg.probability(program)
        assert p <= last
        last = p


def test_unicity_bucketSearch() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    seen = set()
    i = 0
    for program in enumerate_bucket_pcfg(pcfg):
        i = i + 1
        assert program not in seen
        seen.add(program)
    assert len(seen) == cfg.size()


def test_order_bucketSearch() -> None:
    dsl = DSL(syntax)
    max_depth = 3
    cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
    pcfg = ConcretePCFG.uniform(cfg)
    last = Bucket()
    for program in enumerate_bucket_pcfg(pcfg):
        p = pcfg.bucket_tuple(program)
        assert p <= last or last == Bucket()
        last = p
