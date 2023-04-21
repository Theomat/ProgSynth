from synth.syntax.type_system import (
    STRING,
    INT,
    BOOL,
    FixedPolymorphicType,
    PolymorphicType,
    List,
    Arrow,
    PrimitiveType,
    Type,
    UnknownType,
    match,
    EmptyList,
)
from typing import List as TList, Set, Tuple
import random


def test_hash() -> None:
    variations: TList[Type] = [INT, BOOL]

    last_level: TList[Type] = variations[:]
    for _ in range(2):
        next_level: TList[Type] = []
        for el in last_level:
            next_level.append(List(el))
            for el2 in last_level:
                next_level.append(Arrow(el, el2))
        last_level = next_level
        variations += next_level

    for i, t1 in enumerate(variations):
        for j, t2 in enumerate(variations):
            if i == j:
                assert hash(t1) == hash(t2)
            else:
                assert hash(t1) != hash(t2)


def test_match() -> None:
    variations: TList[Type] = [INT, BOOL]

    last_level: TList[Type] = variations[:]
    for _ in range(2):
        next_level: TList[Type] = []
        for el in last_level:
            next_level.append(List(el))
            for el2 in last_level:
                next_level.append(Arrow(el, el2))
        last_level = next_level
        variations += next_level

    for i, t1 in enumerate(variations):
        for j, t2 in enumerate(variations):
            if i == j:
                assert match(t1, t2)
            else:
                assert not match(t1, t2)
        assert match(PolymorphicType("a"), t1)
        assert match(t1, PolymorphicType("a"))

        if isinstance(t1, List):
            assert match(t1, List(PolymorphicType("a")))
            assert match(List(PolymorphicType("a")), t1)
        elif isinstance(t1, Arrow):
            assert match(t1, Arrow(PolymorphicType("a"), t1.type_out))
            assert match(t1, Arrow(t1.type_in, PolymorphicType("a")))
            assert match(Arrow(PolymorphicType("a"), t1.type_out), t1)
            assert match(Arrow(t1.type_in, PolymorphicType("a")), t1)


def test_decompose_type() -> None:
    variations: TList[Tuple[Type, Set[PrimitiveType], Set[PolymorphicType]]] = [
        (INT, set([INT]), set()),
        (BOOL, set([BOOL]), set()),
        (PolymorphicType("a"), set(), set([PolymorphicType("a")])),
        (PolymorphicType("b"), set(), set([PolymorphicType("b")])),
    ]

    last_level = variations[:]
    for _ in range(2):
        next_level: TList[Tuple[Type, Set[PrimitiveType], Set[PolymorphicType]]] = []
        for el, sb, sp in last_level:
            next_level.append((List(el), sb, sp))
            for el2, sb2, sp2 in last_level:
                next_level.append((Arrow(el, el2), sb | sb2, sp | sp2))
        last_level = next_level
        variations += next_level

    for el, sb, sp in variations:
        a, b = el.decompose_type()
        assert a == sb
        assert b == sp


def test_unify() -> None:
    t = Arrow(
        Arrow(INT, PolymorphicType("a")),
        List(Arrow(PolymorphicType("b"), PolymorphicType("a"))),
    )
    assert t.unify({"a": BOOL, "b": STRING}) == Arrow(
        Arrow(INT, BOOL), List(Arrow(STRING, BOOL))
    )


def test_contains() -> None:
    t = Arrow(
        Arrow(INT, EmptyList),
        List(Arrow(PolymorphicType("b"), PolymorphicType("a"))),
    )
    assert EmptyList in t
    assert BOOL not in t
    assert PolymorphicType("b") in t
    assert Arrow(INT, EmptyList) in t


def test_is_a() -> None:
    types = [BOOL, INT, List(INT), Arrow(INT, INT)]
    for i, x in enumerate(types):
        for y in types[i + 1 :]:
            print("x:", x)
            print("y:", y)
            assert not x.is_a(y)
            assert not y.is_a(x)
            assert x.is_a(x | y)
            assert y.is_a(x | y)
            assert not (x | y).is_a(x)
            assert (x | y).is_a(x | y)

    assert List(INT).is_a(List(PolymorphicType("a")))
    assert List(INT).is_a(List(INT | BOOL))
    assert not List(STRING).is_a(List(INT | BOOL))


def test_can_be() -> None:
    z = FixedPolymorphicType("z", INT, BOOL)
    types = [BOOL, INT, List(INT), Arrow(INT, INT)]
    for i, x in enumerate(types):
        assert z.can_be(x) == (i <= 1)
