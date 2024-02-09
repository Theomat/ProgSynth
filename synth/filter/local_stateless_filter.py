from typing import Callable, Dict, Generic, TypeVar

from synth.filter.filter import Filter
from synth.syntax.program import Function, Program, Primitive

V = TypeVar("V")


class LocalStatelessFilter(Filter, Generic[V]):
    def __init__(self, should_reject: Dict[str, Callable]) -> None:
        self.should_reject = should_reject

    def accept(self, program: Program) -> bool:
        accepted = True
        if isinstance(program, Function):
            fun: Primitive = program.function  # type: ignore
            rejects = self.should_reject.get(fun.primitive, None)
            accepted = rejects is None or not rejects(*program.arguments)
        return accepted


def commutative_rejection(p1: Program, p2: Program) -> bool:
    """
    Rejection filter to have unique programs for a commutative binary operator
    """
    return hash(p1) <= hash(p2)


def reject_functions(p: Program, *function_names: str) -> bool:
    """
    Rejects any function whose name is in the parameters
    """
    if isinstance(p, Function):
        return p.function.primitive in function_names  # type: ignore
    return False
