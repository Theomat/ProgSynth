from synth.syntax.grammars.cfg import CFG
from synth.syntax.dsl import DSL
from synth.syntax.type_system import BOOL, INT, FunctionType
from synth.tools.type_constraints.pattern_constraints import (
    produce_new_syntax_for_constraints,
)


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "*": FunctionType(INT, INT, INT),
    "<=": FunctionType(INT, INT, BOOL),
    "==": FunctionType(INT, INT, BOOL),
    "and": FunctionType(BOOL, BOOL, BOOL),
    "or": FunctionType(BOOL, BOOL, BOOL),
    "not": FunctionType(BOOL, BOOL),
    "ite": FunctionType(BOOL, INT, INT, INT),
    "0": INT,
    "1": INT,
    "2": INT,
}
constraints = [
    "and ^and *",
    "or ^or,and ^and",
    "+ ^+,0 ^0",
    "+ $(var0) *",
    "not ^not,and",
    "* ^*,0,1 ^0,1",
    "- * ^0",
]
depth = 5
type_request = FunctionType(INT, INT, INT)


def test_produce() -> None:
    old_size = -1

    for _ in range(4):
        new_syntax, new_tr = produce_new_syntax_for_constraints(
            syntax, constraints, type_request, progress=False
        )
        size = CFG.depth_constraint(DSL(new_syntax), new_tr, depth).size()
        if old_size == -1:
            old_size = size
        assert old_size == size


def test_constraints() -> None:
    dsl = DSL(syntax)
    p1 = dsl.parse_program("(+ var1 1)", type_request)
    p2 = dsl.parse_program("(- var0 0)", type_request)
    p3 = dsl.parse_program("(+ var0 0)", type_request)
    new_syntax, new_tr = produce_new_syntax_for_constraints(
        syntax, constraints, type_request, progress=False
    )
    cfg = CFG.depth_constraint(DSL(new_syntax), new_tr, depth)
    assert p1 not in cfg
    assert p2 not in cfg
    assert p3 not in cfg
