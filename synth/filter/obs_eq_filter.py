from typing import Any, Dict, List, Tuple

from synth.filter.filter import Filter
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program


class ObsEqFilter(Filter):
    def __init__(self, evaluator: Evaluator, inputs_list: List[List[Any]]) -> None:
        self.evaluator = evaluator
        self.inputs_list = inputs_list
        self._cache: Dict[Tuple[Any, ...], Program] = {}

    def _eval(self, prog: Program) -> bool:
        """
        Returns True iff the prog is unique wrt to outputs
        """
        outputs = None
        for inputs in self.inputs_list:
            out = self.evaluator.eval(prog, inputs)
            if out is None:
                return False
            outputs = (outputs, out)
        original = self._cache.get(outputs)  # type: ignore
        if original is not None:
            return False
        else:
            self._cache[outputs] = prog  # type: ignore
            return True

    def accept(self, obj: Program) -> bool:
        return self._eval(obj)

    def reset_cache(self) -> None:
        self._cache.clear()
