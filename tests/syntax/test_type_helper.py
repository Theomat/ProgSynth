from synth.syntax.type_system import (
    STRING,
    INT,
    BOOL,
    FixedPolymorphicType,
    GenericFunctor,
    PolymorphicType,
    List,
    Arrow,
    PrimitiveType,
    UnknownType,
    match,
)
from synth.syntax.type_helper import auto_type, FunctionType, guess_type
import random


def test_guess_type() -> None:
    # Bool
    assert guess_type(True) == BOOL
    assert guess_type(False) == BOOL
    # Int
    random.seed(0)
    for _ in range(100):
        assert guess_type(random.randint(-100, 100)) == INT
    # String
    assert guess_type("") == STRING
    # List
    assert match(guess_type([]), List(PolymorphicType("")))
    assert guess_type([True]) == List(BOOL)
    assert guess_type([""]) == List(STRING)
    assert guess_type([1]) == List(INT)
    # Unknown
    assert isinstance(guess_type(int), UnknownType)


def test_FunctionType() -> None:
    assert FunctionType(INT, BOOL, STRING, List(INT)) == Arrow(
        INT, Arrow(BOOL, Arrow(STRING, List(INT)))
    )


def test_auto_type_base() -> None:
    assert PrimitiveType("int") == auto_type("int")
    assert PrimitiveType("bb") == auto_type("bb")
    assert PolymorphicType("bb") == auto_type("'bb")
    assert PolymorphicType("aa") == auto_type("'aa")


def test_auto_type_advanced() -> None:
    assert List(PrimitiveType("int")) == auto_type("int list")
    assert List(PolymorphicType("a")) == auto_type("'a list")

    some = GenericFunctor("some", min_args=1, max_args=1)
    opt = GenericFunctor("optional", min_args=1, max_args=1)

    assert some(PolymorphicType("a")) == auto_type("'a some")
    assert opt(PolymorphicType("a")) == auto_type("'a optional")
    assert opt(some(PolymorphicType("a"))) == auto_type("'a some optional")

    x = PrimitiveType("bb") | PolymorphicType("aa")
    assert x == auto_type("bb | 'aa")
    assert x == auto_type("bb|'aa")
    assert x == auto_type("'aa | bb")


def test_auto_type_arrows() -> None:
    a = PrimitiveType("a")
    b = PrimitiveType("b")
    assert FunctionType(a, b) == auto_type("a->b")
    assert FunctionType(a, b, b) == auto_type("a->b->b")
    assert FunctionType(a, FunctionType(a, b), b) == auto_type("a->(a->b)->b")


def test_auto_type_fixed_poly() -> None:
    x = FixedPolymorphicType("z", PrimitiveType("b") | PrimitiveType("c"))
    assert x == auto_type("'z[b|c]")
    assert x == auto_type("'z [b|c]")
    assert x == auto_type("'z[ b|c ]")
