from typing import Callable, Dict, Set, Tuple

from synth.filter.filter import Filter
from synth.syntax.program import Function, Primitive, Program
from synth.syntax.type_system import Arrow, Type


SyntacticFilter = Filter[Tuple[Type, Program]]


class UseAllVariablesFilter(SyntacticFilter):
    def __init__(self) -> None:
        super().__init__()
        self._cached_variables_set: Dict[Type, Set[int]] = {}

    def __get_var_set__(self, treq: Type) -> Set[int]:
        if treq not in self._cached_variables_set:
            if treq.is_instance(Arrow):
                self._cached_variables_set[treq] = set(range(len(treq.arguments())))
            else:
                self._cached_variables_set[treq] = set()

        return self._cached_variables_set[treq]

    def accept(self, obj: Tuple[Type, Program]) -> bool:
        treq, prog = obj
        target = self.__get_var_set__(treq)
        return prog.used_variables() == target


class FunctionFilter(SyntacticFilter):
    def __init__(self, is_useless: Dict[str, Callable]) -> None:
        super().__init__()
        self.is_useless = is_useless

    def accept(self, obj: Tuple[Type, Program]) -> bool:
        _, prog = obj
        for P in prog.depth_first_iter():
            if not isinstance(P, Function):
                continue
            f = P.function
            if not isinstance(f, Primitive):
                continue
            if f.primitive in self.is_useless and self.is_useless[f.primitive](
                *P.arguments
            ):
                return False
        return True


class SetFilter(SyntacticFilter):
    def __init__(self, forbidden: Set[Program]) -> None:
        super().__init__()
        self.forbidden = forbidden

    def accept(self, obj: Tuple[Type, Program]) -> bool:
        return obj[1] not in self.forbidden
