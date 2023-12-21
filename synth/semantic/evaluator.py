from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

from synth.syntax.program import Constant, Function, Primitive, Program, Variable


class Evaluator(ABC):
    @abstractmethod
    def eval(self, program: Program, input: Any) -> Any:
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """
        Clear any cache this evaluator might use.
        """
        pass


def __tuplify__(element: Any) -> Any:
    if isinstance(element, List):
        return tuple(__tuplify__(x) for x in element)
    else:
        return element


class DSLEvaluator(Evaluator):
    def __init__(self, semantics: Dict[Primitive, Any], use_cache: bool = True) -> None:
        super().__init__()
        self.semantics = semantics
        self.use_cache = use_cache
        self._cache: Dict[Any, Dict[Program, Any]] = {}
        self._cons_cache: Dict[Any, Dict[Program, Any]] = {}
        self.skip_exceptions: Set[Exception] = set()
        # Statistics
        self._total_requests = 0
        self._cache_hits = 0

    def eval(self, program: Program, input: List) -> Any:
        key = __tuplify__(input)
        if self.use_cache and key not in self._cache:
            self._cache[key] = {}
        evaluations: Dict[Program, Any] = self._cache[key] if self.use_cache else {}
        if program in evaluations:
            return evaluations[program]
        try:
            for sub_prog in program.depth_first_iter():
                self._total_requests += 1
                if sub_prog in evaluations:
                    self._cache_hits += 1
                    continue
                if isinstance(sub_prog, Primitive):
                    evaluations[sub_prog] = self.semantics[sub_prog]
                elif isinstance(sub_prog, Variable):
                    evaluations[sub_prog] = input[sub_prog.variable]
                elif isinstance(sub_prog, Constant):
                    evaluations[sub_prog] = sub_prog.value
                elif isinstance(sub_prog, Function):
                    fun = evaluations[sub_prog.function]
                    for arg in sub_prog.arguments:
                        fun = fun(evaluations[arg])
                    evaluations[sub_prog] = fun
        except Exception as e:
            if type(e) in self.skip_exceptions:
                evaluations[program] = None
                return None
            else:
                raise e

        return evaluations[program]

    def clear_cache(self) -> None:
        self._cache = {}
        self._cons_cache = {}

    @property
    def cache_hit_rate(self) -> float:
        return self._cache_hits / self._total_requests
