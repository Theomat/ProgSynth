from abc import ABC, abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Any

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

    def _init_stats_(self) -> None:
        self._stats["programs"] = 0
        self._stats["time"] = 0
        self._stats["program_probability"] = 0

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

    def _init_task_solving_(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> None:
        self._programs = 0

    def _close_task_solving_(
        self,
        task: Task[PBE],
        enumerator: HSEnumerator,
        time_used: float,
        solution: bool,
        last_program: Program,
    ) -> None:
        self._stats["time"] += time_used
        self._stats["program_probability"] = enumerator.G.probability(last_program)
        self._stats["program"] += self._programs

    def solve(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> Generator[Program, bool, None]:
        with chrono.clock(f"solve.{self.name()}") as c:  # type: ignore
            self._init_task_solving_(task, enumerator, timeout)
            for program in enumerator:
                time = c.elapsed_time()
                if time >= timeout:
                    self._close_task_solving_(task, enumerator, time, False, program)
                    return
                self._programs += 1
                if self._test_(task, program):
                    should_stop = yield program
                    if should_stop:
                        self._close_task_solving_(task, enumerator, time, True, program)
                        return

    def _test_(self, task: Task[PBE], program: Program) -> bool:
        failed = False
        for ex in task.specification.examples:
            if self.evaluator.eval(program, ex.inputs) != ex.output:
                failed = True
        return not failed


class NaivePBESolver(PBESolver):
    @classmethod
    def name(cls) -> str:
        return "naive"


class MetaPBESolver(PBESolver, ABC):
    def __init__(
        self,
        evaluator: DSLEvaluator,
        solver_builder: Callable[..., PBESolver] = NaivePBESolver,
        **kwargs: Any,
    ) -> None:
        self.subsolver = solver_builder(evaluator, **kwargs)
        self.evaluator = evaluator
        self._stats: Dict[str, Any] = {}
        self._init_stats_()

    def _init_stats_(self) -> None:
        super()._init_stats_()
        self.subsolver._init_stats_()
        for name, val in self.subsolver._stats.items():
            if name not in self._stats:
                self._stats[name] = val

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    def reset_stats(self) -> None:
        super().reset_stats()
        self.subsolver.reset_stats()

    def _init_task_solving_(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> None:
        self.subsolver._init_task_solving_(task, enumerator, timeout)

    def _close_task_solving_(
        self,
        task: Task[PBE],
        enumerator: HSEnumerator,
        time_used: float,
        solution: bool,
        last_program: Program,
    ) -> None:
        self.subsolver._close_task_solving_(
            task, enumerator, time_used, solution, last_program
        )
        for name, val in self.subsolver._stats.items():
            self._stats[name] = val
        super()._close_task_solving_(
            task, enumerator, time_used, solution, last_program
        )

    def _test_(self, task: Task[PBE], program: Program) -> bool:
        return self.subsolver._test_(task, program)


class CutoffPBESolver(PBESolver):
    """
    A solver that fails a program on first fail.
    """

    @classmethod
    def name(cls) -> str:
        return "cutoff"

    def _test_(self, task: Task[PBE], program: Program) -> bool:
        for ex in task.specification.examples:
            if self.evaluator.eval(program, ex.inputs) != ex.output:
                return False
        return True


class ObsEqPBESolver(PBESolver):
    """
    A solver that use observational equivalence.
    """

    @classmethod
    def name(cls) -> str:
        return "obs-eq"

    def _init_stats_(self) -> None:
        super()._init_stats_()
        self._stats["merged"] = 0

    def _init_task_solving_(
        self, task: Task[PBE], enumerator: HSEnumerator, timeout: float = 60
    ) -> None:
        super()._init_task_solving_(task, enumerator, timeout)
        self._merged = 0
        self._results: Dict[Any, Any] = {}
        self._enumerator = enumerator

    def _close_task_solving_(
        self,
        task: Task[PBE],
        enumerator: HSEnumerator,
        time_used: float,
        solution: bool,
        last_program: Program,
    ) -> None:
        super()._close_task_solving_(
            task, enumerator, time_used, solution, last_program
        )
        self._stats["merged"] += self._merged

    def _test_(self, task: Task[PBE], program: Program) -> bool:
        failed = False
        outputs = None
        for ex in task.specification.examples:
            out = self.evaluator.eval(program, ex.inputs)
            failed |= out != ex.output
            if isinstance(out, list):
                outputs = (outputs, tuple(out))
            else:
                outputs = (outputs, out)  # type: ignore

        if failed:
            original = self._results.get(outputs)
            if original is not None:
                self._enumerator.merge_program(original, program)
                self._merged += 1
            else:
                self._results[outputs] = program
        return not failed
