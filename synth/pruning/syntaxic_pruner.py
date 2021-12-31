from typing import Callable, Dict, Set, Tuple

from synth.pruning.pruner import Pruner
from synth.syntax.program import Function, Primitive, Program
from synth.syntax.type_system import Arrow, Type


SyntaxicPruner = Pruner[Tuple[Type, Program]]


class UseAllVariablesPruner(SyntaxicPruner):
    def __init__(self) -> None:
        super().__init__()
        self._cached_variables_set: Dict[Type, Set[int]] = {}

    def __get_var_set__(self, treq: Type) -> Set[int]:
        if treq not in self._cached_variables_set:
            if isinstance(treq, Arrow):
                self._cached_variables_set[treq] = set(range(len(treq.arguments())))
            else:
                self._cached_variables_set[treq] = set()

        return self._cached_variables_set[treq]

    def accept(self, obj: Tuple[Type, Program]) -> bool:
        treq, prog = obj
        target = self.__get_var_set__(treq)
        return prog.used_variables() == target


class FunctionPruner(SyntaxicPruner):
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
