from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import PrimitiveType


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


class DSLEvaluatorWithConstant(Evaluator):
    def __init__(
        self,
        semantics: Dict[Primitive, Any],
        constant_types: Set[PrimitiveType],
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.semantics = semantics
        self.constant_types = constant_types
        self.use_cache = use_cache
        self._cache: Dict[Any, Dict[Program, Any]] = {}
        self._cons_cache: Dict[Any, Dict[Program, Any]] = {}
        self._invariant_cache: Dict[Program, Any] = {}
        self.skip_exceptions: Set[Exception] = set()
        # Statistics
        self._total_requests = 0
        self._cache_hits = 0

    def eval_with_constant(
        self, program: Program, input: List, constant_in: str, constant_out: str
    ) -> Any:
        evaluations: Dict[Program, Any] = {}
        if self.use_cache:
            used_cons = False
            for sub_prog in program.depth_first_iter():
                if (
                    isinstance(sub_prog, Primitive)
                    and sub_prog.type in self.constant_types
                ):
                    used_cons = True
                    break
            if used_cons:
                key = input.copy()
                key.append(constant_in)
                key.append(constant_out)
                key = __tuplify__(key)
                evaluations = self._cons_cache[key] if key in self._cons_cache else {}
            else:
                key = __tuplify__(input)
                evaluations = self._cache[key] if key in self._cache else {}

        if program in evaluations:
            return evaluations[program]
        try:
            for sub_prog in program.depth_first_iter():
                self._total_requests += 1
                if sub_prog in evaluations:
                    self._cache_hits += 1
                    continue
                if sub_prog.is_invariant(self.constant_types):
                    if sub_prog in self._invariant_cache:
                        self._cache_hits += 1
                        evaluations[sub_prog] = self._invariant_cache[sub_prog]
                        continue
                    else:
                        self._invariant_cache[sub_prog] = None
                if isinstance(sub_prog, Primitive):
                    if sub_prog.primitive == "cst_in":
                        evaluations[sub_prog] = constant_in
                    elif sub_prog.primitive == "cst_out":
                        evaluations[sub_prog] = constant_out
                    else:
                        evaluations[sub_prog] = self.semantics[sub_prog]
                elif isinstance(sub_prog, Variable):
                    evaluations[sub_prog] = input[sub_prog.variable]
                elif isinstance(sub_prog, Function):
                    fun = evaluations[sub_prog.function]
                    for arg in sub_prog.arguments:
                        fun = fun(evaluations[arg])
                    evaluations[sub_prog] = fun
                if sub_prog.is_invariant(self.constant_types):
                    self._invariant_cache[sub_prog] = evaluations[sub_prog]

        except Exception as e:
            if type(e) in self.skip_exceptions:
                evaluations[program] = None
                return None
            else:
                print(e)
                raise e

        return evaluations[program]

    def eval(self, program: Program, input: List) -> Any:
        if len(input) >= 3:
            return self.eval_with_constant(program, input[2:], input[0], input[1])
        return self.eval_with_constant(program, input, "", "")

    def clear_cache(self) -> None:
        self._cache = {}
        self._cons_cache = {}
        self._invariant_cache = {}

    @property
    def cache_hit_rate(self) -> float:
        return self._cache_hits / self._total_requests
