from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Variable
from synth.syntax.type_system import (
    INT,
    STRING,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.pruning.constraints.parsing import (
    parse_specification,
    TokenAnything,
    TokenAllow,
    TokenAtLeast,
    TokenAtMost,
    TokenFunction,
    TokenVarDep,
)


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
cfg = TTCFG.size_constraint(dsl, FunctionType(INT, INT), 20)


def test_bases() -> None:
    assert parse_specification("_", cfg) == TokenAnything()
    assert parse_specification("#(1)<=3", cfg), TokenAtMost(
        [dsl.get_primitive("1")], count=3
    )
    assert parse_specification("#(1,+)>=3", cfg), TokenAtLeast(
        [dsl.get_primitive("1"), dsl.get_primitive("+")], count=4
    )
    assert parse_specification("#(1,+,+)>=3", cfg), TokenAtLeast(
        [dsl.get_primitive("1"), dsl.get_primitive("+")], count=4
    )
    assert parse_specification("(+ 1 _)", cfg) == TokenFunction(
        TokenAllow([dsl.get_primitive("+")]),
        [TokenAllow([dsl.get_primitive("1")]), TokenAnything()],
    )
    assert parse_specification("$(var0)", cfg), TokenVarDep([Variable(0, INT)])
