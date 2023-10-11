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
        with chrono.clock("search.naive") as c:  # type: ignore
            programs = 0
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._stats["time"] += time
                    self._stats["program_probability"] = enumerator.G.probability(
                        program
                    )
                    self._stats["program"] += programs
                    return
                programs += 1
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
                        self._stats["program"] += programs
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
        with chrono.clock("search.cutoff") as c:  # type: ignore
            programs = 0
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._stats["time"] += time
                    self._stats["program_probability"] = enumerator.G.probability(
                        program
                    )
                    self._stats["program"] += programs
                    return
                programs += 1
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
                        self._stats["program"] += programs
                        return


class ObsEqPBESolver(PBESolver):
    """
    A solver that use observational equivalence.
    """

    @classmethod
    def name(cls) -> str:
        return "obs-eq"

    def _init_stats_(self) -> None:
        self._stats["programs"] = 0
        self._stats["time"] = 0
        self._stats["program_probability"] = 0
        self._stats["merged"] = 0

    def solve(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> Generator[Program, bool, None]:
        with chrono.clock("search.obs-eq") as c:  # type: ignore
            results: Dict[Any, Any] = {}
            merged = 0
            programs = 0
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._stats["merged"] += merged
                    self._stats["time"] += time
                    self._stats["program_probability"] = enumerator.G.probability(
                        program
                    )
                    self._stats["program"] += programs
                    return
                programs += 1
                failed = False
                outputs = None
                for ex in task.specification.examples:
                    out = self.evaluator.eval(program, ex.inputs)
                    failed |= out != ex.output
                    if isinstance(out, list):
                        outputs = (outputs, tuple(out))
                    else:
                        outputs = (outputs, out)  # type: ignore
                if not failed:
                    should_stop = yield program
                    if should_stop:
                        self._stats["time"] += time
                        self._stats["program_probability"] = enumerator.G.probability(
                            program
                        )
                        self._stats["merged"] += merged
                        self._stats["program"] += programs
                        return
                else:
                    original = results.get(outputs)
                    if original is not None:
                        enumerator.merge_program(original, program)
                        merged += 1
                    else:
                        results[outputs] = program
