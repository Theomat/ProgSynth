from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType
from synth.filter.constraints.ttcfg_constraints import add_constraints


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), 9)
# cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), 4)


def test_restriction() -> None:
    new_cfg = add_constraints(cfg, [], "(+ 1 _)", progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) in new_cfg

    new_cfg = add_constraints(cfg, ["(+ 1 _)"], progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) in new_cfg


def test_multi_level() -> None:
    new_cfg = add_constraints(cfg, [], "(+ 1 (+ _ 1))", progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg

    new_cfg = add_constraints(cfg, ["(+ 1 (+ _ 1))"], progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) not in new_cfg


def test_at_most() -> None:
    new_cfg = add_constraints(cfg, [], "(- #(1)<=1 _)", progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) not in new_cfg

    new_cfg = add_constraints(cfg, ["(- #(1)<=1 _)"], progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg


def test_var_dep() -> None:
    new_cfg = add_constraints(cfg, [], "(+ >^(var0) _)", progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) in new_cfg

    new_cfg = add_constraints(cfg, ["(+ >^(var0) _)"], progress=False)
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) not in new_cfg
