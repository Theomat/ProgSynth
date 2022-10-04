from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.pruning.constraints.dfta_constraints import add_dfta_constraints


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
cfg = CFG.depth_constraint(dsl, FunctionType(INT, INT), 4)


def test_restriction() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ 1 _)"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ 1 _)"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) in new_cfg


def test_multi_level() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ 1 (+ _ 1))"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ 1 (+ _ 1))"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(+ 1 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (- 1 1) 1))", cfg.type_request) in new_cfg


def test_at_most() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(- #(1)<=1 _)"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) not in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(- #(1)<=1 _)"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ (+ 1 1) 1))", cfg.type_request) in new_cfg


def test_at_least() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(- _ #(1)>=2)"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- (+ 1 1) 1))", cfg.type_request) in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(- _ #(1)>=2)"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- (- 1 1) 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (- 1 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 (+ 1 1)))", cfg.type_request) in new_cfg


def test_forbid_subtree() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ >^(var0) _)"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ >^(var0) _)"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) not in new_cfg


def test_force_subtree() -> None:
    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ >(var0) _)"], sketch=True, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ (+ 1 (+ var0 1)) 1)", cfg.type_request) in new_cfg

    new_cfg = UCFG.from_DFTA(
        add_dfta_constraints(cfg, ["(+ >(var0) _)"], sketch=False, progress=False)
    )
    print(new_cfg)
    assert dsl.parse_program("(- 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(- 1 (- 1 1))", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 1))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ var0 1)", cfg.type_request) in new_cfg
    assert dsl.parse_program("(+ 1 (+ 1 (+ var0 1)))", cfg.type_request) not in new_cfg
    assert dsl.parse_program("(+ (+ (+ var0 1) 1) 1)", cfg.type_request) in new_cfg