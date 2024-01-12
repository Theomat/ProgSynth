from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Variable
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType
from synth.filter.constraints.parsing import (
    parse_specification,
    TokenAnything,
    TokenAllow,
    TokenAtLeast,
    TokenAtMost,
    TokenFunction,
    TokenForceSubtree,
    TokenForbidSubtree,
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
ONE = dsl.get_primitive("1")
PLUS = dsl.get_primitive("+")


def test_bases() -> None:
    assert parse_specification("_", cfg) == TokenAnything()
    assert parse_specification("#(1)<=3", cfg) == TokenAtMost([ONE], count=3)
    assert parse_specification("(+ #(1)<=1 _)", cfg) == TokenFunction(
        TokenAllow([PLUS]), args=[TokenAtMost([ONE], count=1), TokenAnything()]
    )
    assert parse_specification("#(1,+)>=3", cfg) == TokenAtLeast([ONE, PLUS], count=3)
    assert parse_specification("#(1,+,+)>=4", cfg) == TokenAtLeast([ONE, PLUS], count=4)
    assert parse_specification("(+ 1 _)", cfg) == TokenFunction(
        TokenAllow([PLUS]),
        [TokenAllow([ONE]), TokenAnything()],
    )
    assert parse_specification(">(var0)", cfg) == TokenForceSubtree([Variable(0, INT)])
    assert parse_specification(">^(1,var0)", cfg) == TokenForbidSubtree(
        [ONE, Variable(0, INT)]
    )
