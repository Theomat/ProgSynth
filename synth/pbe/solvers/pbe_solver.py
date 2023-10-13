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
        """
        Returns the name of the class of this solver.
        """
        pass

    def full_name(self) -> str:
        """
        Returns the name of this particular instance of the solver, it may contain additional information.
        """
        return self.name()

    def reset_stats(self) -> None:
        """
        Reset the statistics collected by this solver.
        """
        self._stats = {}
        self._init_stats_()

    def get_stats(self, name: str) -> Optional[Any]:
        """
        Get a stat by name or None if it does not exist.
        """
        return self._stats.get(name, None)

    def available_stats(self) -> List[str]:
        """
        List the name of all currently avialable stats.
        """
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
        """
        Solve the given task by enumerating programs with the given enumerator.
        When the timeout is reached, this function returns.
        When a program that satisfies the task has been found, yield it.
        The calling function should then send True if and only if it accepts the solution.
        If False is sent the search continues.
        """
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
        """
        Return true iff program satisfies the specification given by the task.
        Fills self._score with a score representing how close the program was to solve the task.

        POSTCOND:
            0 <= self._score <= 1
        """
        failed = False
        success = 0
        for ex in task.specification.examples:
            if self.evaluator.eval(program, ex.inputs) != ex.output:
                failed = True
            else:
                success += 1
        self._score = success / len(task.specification.examples)
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

    def full_name(self) -> str:
        """
        Returns the name of this particular instance of the solver, for meta solver it has the format <solver_full_name>.<sub_solver_full_name>.
        """
        return super().full_name() + "." + self.subsolver.full_name()

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
    A solver that fails a program on first example that fails.
    """

    @classmethod
    def name(cls) -> str:
        return "cutoff"

    def _test_(self, task: Task[PBE], program: Program) -> bool:
        n = 0
        for ex in task.specification.examples:
            if self.evaluator.eval(program, ex.inputs) != ex.output:
                self._score = n / len(task.specification.examples)
                return False
            n += 1
        self._score = 1
        return True


class ObsEqPBESolver(PBESolver):
    """
    A solver that uses observational equivalence.
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
        success = 0
        for ex in task.specification.examples:
            out = self.evaluator.eval(program, ex.inputs)
            local_success = out == ex.output
            failed |= not local_success
            success += local_success
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

        self._score = success / len(task.specification.examples)
        return not failed
