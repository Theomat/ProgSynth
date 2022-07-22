from collections import defaultdict
from typing import Set, Tuple
from synth.pruning.type_constraints.utils import get_prefix
from synth.syntax.grammars.cfg import CFG, CFGState
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive
from synth.syntax.type_system import INT, FunctionType, Type
from synth.pruning.type_constraints.pattern_constraints import (
    produce_new_syntax_for_constraints,
)


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "*": FunctionType(INT, INT, INT),
    "0": INT,
    "1": INT,
    "2": INT,
}
constraints = ["+ $(var0) _", "- (+ (+ _ _) _) _", "* ^+,-,* _"]
depth = 4
type_request = FunctionType(INT, INT, INT)


def test_produce() -> None:
    old_size = -1

    for _ in range(2):
        new_syntax, new_tr = produce_new_syntax_for_constraints(
            syntax, constraints, type_request, progress=True
        )
        size = CFG.depth_constraint(DSL(new_syntax), new_tr, depth).size()
        if old_size == -1:
            old_size = size
        assert old_size == size


def test_var_constraints() -> None:
    dsl = DSL(syntax)
    p1 = dsl.parse_program("(+ var1 1)", type_request)
    p2 = dsl.parse_program("(- var0 0)", type_request)
    p3 = dsl.parse_program("(+ var0 0)", type_request)
    new_syntax, new_tr = produce_new_syntax_for_constraints(
        syntax, constraints[:1], type_request, progress=False
    )
    cfg = CFG.depth_constraint(DSL(new_syntax), new_tr, depth)
    assert p1 not in cfg
    assert p2 not in cfg
    assert p3 not in cfg


def test_forbid_constraints() -> None:
    dsl = DSL(syntax)
    p1 = dsl.parse_program("(* (+ 0 1) 1)", type_request)
    p2 = dsl.parse_program("(* 0 (+ 1 1))", type_request)
    p3 = dsl.parse_program("(* 2 (* (* 1 2) 2))", type_request)
    new_syntax, new_tr = produce_new_syntax_for_constraints(
        syntax, constraints[2:], type_request, progress=False
    )
    cfg = CFG.depth_constraint(DSL(new_syntax), new_tr, depth)
    p2 = cfg.embed(p2)
    assert p1 not in cfg
    assert p2 in cfg
    assert p3 not in cfg


def test_nested_constraints() -> None:
    dsl = DSL(syntax)
    p1 = dsl.parse_program("(- (+ 0 1) 1)", type_request)
    p2 = dsl.parse_program("(- 0 (+ (+ 1 1) 1))", type_request)
    p3 = dsl.parse_program("(- (+ (+ 1 2) 2) 2)", type_request)
    new_syntax, new_tr = produce_new_syntax_for_constraints(
        syntax, constraints[1:2], type_request, progress=False
    )
    cfg = CFG.depth_constraint(DSL(new_syntax), new_tr, depth)
    assert p1 not in cfg
    assert p2 not in cfg
    assert p3 not in cfg


def __exist_equivalent_path_from_start__(cfg: CFG, pset: Set[Primitive]) -> bool:
    plist = list(pset)
    r = cfg.rules[cfg.start]
    for i, p1 in enumerate(plist):
        for p2 in plist[i + 1 :]:
            if all(
                __exist_equivalent_path__(cfg, arg1, arg2)
                for arg1, arg2 in zip(r[p1][0], r[p2][0])
            ):
                print(f"\tstart -> {p1} and start -> {p2}")
                return True
    return False


def __exist_equivalent_path__(
    cfg: CFG, s1: Tuple[Type, CFGState], s2: Tuple[Type, CFGState]
) -> bool:
    if s1 == s2:
        return True
    r1 = cfg.rules[(s1[0], (s1[1], None))]
    r2 = cfg.rules[(s2[0], (s2[1], None))]
    for P1 in r1:
        for P2 in r2:
            if P1 == P2:
                print(f"\t{s1} -> {P1} and {s2} -> {P2}")
                return True
            if (
                isinstance(P1, Primitive)
                and isinstance(P2, Primitive)
                and get_prefix(P1.primitive) == get_prefix(P2.primitive)
            ):
                if all(
                    __exist_equivalent_path__(cfg, arg1, arg2)
                    for arg1, arg2 in zip(r1[P1][0], r2[P2][0])
                ):
                    print(f"\t{s1} -> {P1} and {s2} -> {P2}")
                    return True
    return False


def test_unambiguous() -> None:
    new_syntax, new_tr = produce_new_syntax_for_constraints(
        syntax, constraints[1:2], type_request, progress=False
    )
    cfg = CFG.depth_constraint(DSL(new_syntax), new_tr, depth)
    equivalents = defaultdict(set)
    for P in cfg.rules[cfg.start]:
        if isinstance(P, Primitive):
            equivalents[get_prefix(P.primitive)].add(P)
    for plist in equivalents.values():
        if len(plist) > 1:
            assert not __exist_equivalent_path_from_start__(cfg, plist)
