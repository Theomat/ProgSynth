from typing import Dict, Generic, TypeVar, Optional

from synth.filter.filter import Filter
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Function, Program, Lambda

V = TypeVar("V")


class DFTAFilter(Filter, Generic[V]):
    def __init__(self, dfta: DFTA[V, DerivableProgram]) -> None:
        self.dfta = dfta
        self._cache: Dict[Program, V] = {}

    def _get_prog_state(self, prog: Program) -> Optional[V]:
        state = self._cache.get(prog, None)
        if state is not None:
            return state
        if isinstance(prog, Function):
            fun = prog.function
            args = tuple(self._get_prog_state(arg) for arg in prog.arguments)
            state = self.dfta.read(fun, args)  # type: ignore
            if state is not None:
                self._cache[prog] = state
            return state
        elif isinstance(prog, Lambda):
            assert False, "Not implemented"
        else:
            state = self.dfta.read(prog, ())  # type: ignore
            if state is not None:
                self._cache[prog] = state
            return state

    def accept(self, obj: Program) -> bool:
        return self._get_prog_state(obj) is not None

    def reset_cache(self) -> None:
        self._cache.clear()
