from abc import ABC, abstractmethod
from typing import Any, Dict, List

from synth.syntax.program import Function, Primitive, Program, Variable


class Evaluator(ABC):
    @abstractmethod
    def eval(self, program: Program, input: Any) -> Any:
        pass


class DSLEvaluator(Evaluator):
    def __init__(self, semantics: Dict[str, Any], use_cache: bool = True) -> None:
        super().__init__()
        self.semantics = semantics
        self.use_cache = use_cache
        self._cache: Dict[Any, Dict[Program, Any]] = {}

    def eval(self, program: Program, input: List) -> Any:
        key = tuple(input)
        if key not in self._cache and self.use_cache:
            self._cache[key] = {}
        evaluations: Dict[Program, Any] = self._cache[key] if self.use_cache else {}
        for sub_prog in program.depth_first_iter():
            if sub_prog in evaluations:
                continue
            if isinstance(sub_prog, Primitive):
                evaluations[sub_prog] = self.semantics[sub_prog.primitive]
            elif isinstance(sub_prog, Variable):
                evaluations[sub_prog] = input[sub_prog.variable]
            elif isinstance(sub_prog, Function):
                fun = evaluations[sub_prog.function]
                for arg in sub_prog.arguments:
                    fun = fun(evaluations[arg])
                evaluations[sub_prog] = fun

        return evaluations[program]

    def clear_cache(self) -> None:
        self._cache.clear()
