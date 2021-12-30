from typing import Generator, List
import random

from synth.syntax.program import Primitive, Function, Program, Variable
from synth.syntax.type_system import BOOL, INT, STRING, Arrow, FunctionType


def __gen2list__(g: Generator) -> List:
    out = []
    try:
        while True:
            out.append(next(g))
    except StopIteration:
        return out


def test_guess_function_type() -> None:
    vars: List[Program] = [Variable(i, INT) for i in range(10)]
    random.seed(0)
    for c in range(1, len(vars)):
        sub_vars = vars[:c]
        random.shuffle(sub_vars)
        f = Function(
            Primitive("f", FunctionType(*[INT for _ in range(c + 1)])), sub_vars
        )
        assert f.type == INT
        if c > 1:
            for _ in range(random.randint(1, c - 1)):
                sub_vars.pop()
            f = Function(
                Primitive("f", FunctionType(*[INT for _ in range(c + 1)])), sub_vars
            )
            assert f.type == FunctionType(*[INT for _ in range(c - len(sub_vars) + 1)])


def test_depth_first_iter() -> None:
    i, b, f = (
        Primitive("a", INT),
        Variable(0, BOOL),
        Primitive("f", FunctionType(INT, BOOL, INT)),
    )

    assert __gen2list__(i.depth_first_iter()) == [i]
    assert __gen2list__(b.depth_first_iter()) == [b]
    assert __gen2list__(f.depth_first_iter()) == [f]
    fun = Function(f, [i, b])
    assert __gen2list__(fun.depth_first_iter()) == [f, i, b, fun]
    fun2 = Function(f, [fun, b])
    assert __gen2list__(fun2.depth_first_iter()) == [f, f, i, b, fun, b, fun2]


def test_is_using_all_variables() -> None:
    vars: List[Program] = [Variable(i, INT) for i in range(10)]
    random.seed(0)
    for c in range(1, len(vars)):
        sub_vars = vars[:c]
        random.shuffle(sub_vars)
        f = Function(
            Primitive("f", FunctionType(*[INT for _ in range(c + 1)])), sub_vars
        )
        assert len(f.used_variables()) == c
        sub_vars.pop()
        f = Function(
            Primitive("f", FunctionType(*[INT for _ in range(c + 1)])), sub_vars
        )
        assert len(f.used_variables()) == c - 1
