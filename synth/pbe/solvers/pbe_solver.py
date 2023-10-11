from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional, Any

from synth.semantic.evaluator import DSLEvaluator
from synth.specification import PBE
from synth.syntax.grammars.enumeration.heap_search import HSEnumerator
from synth.syntax.program import Program
from synth.task import Task
from synth.utils import chrono


class PBESolver(ABC):
    def __init__(self, evaluator: DSLEvaluator, **kwargs: Any) -> None:
        self.evaluator = evaluator
        self._stats: Dict[str, Any] = {}
        self._init_stats_()

    @abstractmethod
    def _init_stats_(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    def reset_stats(self) -> None:
        self._stats = {}
        self._init_stats_()

    def get_stats(self, name: str) -> Optional[Any]:
        return self._stats.get(name, None)

    def available_stats(self) -> List[str]:
        return list(self._stats.keys())

    @abstractmethod
    def solve(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> Generator[Program, bool, None]:
        pass


class NaivePBESolver(PBESolver):
    @classmethod
    def name(cls) -> str:
        return "naive"

    def _init_stats_(self) -> None:
        self._stats["programs"] = 0
        self._stats["time"] = 0
        self._stats["program_probability"] = 0

    def solve(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> Generator[Program, bool, None]:
        with chrono.clock("search.base") as c: #type: ignore
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._stats["time"] += time
                    self._stats["program_probability"] = enumerator.G.probability(
                        program
                    )

                    return
                self._stats["programs"] += 1
                failed = False
                for ex in task.specification.examples:
                    if self.evaluator.eval(program, ex.inputs) != ex.output:
                        failed = True
                if not failed:
                    should_stop = yield program
                    if should_stop:
                        self._stats["time"] += time
                        self._stats["program_probability"] = enumerator.G.probability(
                            program
                        )
                        return


class CutoffPBESolver(PBESolver):
    """
    A solver that fails a program on first fail.
    """

    @classmethod
    def name(cls) -> str:
        return "cutoff"

    def _init_stats_(self) -> None:
        self._stats["programs"] = 0
        self._stats["time"] = 0
        self._stats["program_probability"] = 0

    def solve(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> Generator[Program, bool, None]:
        with chrono.clock("search.base") as c: #type: ignore
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._stats["time"] += time
                    self._stats["program_probability"] = enumerator.G.probability(
                        program
                    )

                    return
                self._stats["programs"] += 1
                failed = False
                for ex in task.specification.examples:
                    if self.evaluator.eval(program, ex.inputs) != ex.output:
                        failed = True
                        break
                if not failed:
                    should_stop = yield program
                    if should_stop:
                        self._stats["time"] += time
                        self._stats["program_probability"] = enumerator.G.probability(
                            program
                        )
                        return
