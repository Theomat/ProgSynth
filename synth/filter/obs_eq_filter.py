from collections import defaultdict
from typing import Any, Dict, List, Tuple

from synth.filter.filter import Filter
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program
from synth.syntax.type_system import Type


class ObsEqFilter(Filter):
    def __init__(self, evaluator: Evaluator, inputs_list: List[List[Any]]) -> None:
        self.evaluator = evaluator
        self.inputs_list = inputs_list
        self._cache: Dict[Type, Dict[Tuple[Any, ...], Program]] = defaultdict(dict)

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
        original = self._cache[prog.type].get(outputs)  # type: ignore
        if original is not None and hash(original) != hash(prog):
            return False
        else:
            self._cache[prog.type][outputs] = prog  # type: ignore
            return True

    def accept(self, obj: Program) -> bool:
        return self._eval(obj)

    def reset_cache(self) -> None:
        self._cache.clear()
