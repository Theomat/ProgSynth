from typing import Any, Callable, Generator, List, Tuple
from synth.semantic.evaluator import DSLEvaluator


from synth.specification import PBE
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program
from synth.syntax.type_system import Type
from synth.task import Task
from synth.utils import chrono
from synth.pbe.solvers.pbe_solver import MetaPBESolver, NaivePBESolver, PBESolver


class RestartPBESolver(MetaPBESolver):
    def __init__(
        self,
        evaluator: DSLEvaluator,
        solver_builder: Callable[..., PBESolver] = NaivePBESolver,
        restart_criterion: Callable[["RestartPBESolver"], bool] = lambda self: len(
            self._data
        )
        - self._last_size
        > 10000,
        uniform_prior: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(evaluator, solver_builder, **kwargs)
        self.restart_criterion = restart_criterion
        self._last_size: int = 0
        self.uniform_prior = uniform_prior

    def _init_stats_(self) -> None:
        super()._init_stats_()
        self._stats["restarts"] = 0

    @classmethod
    def name(cls) -> str:
        return "restart"

    def _init_task_solving_(
        self, task: Task[PBE], enumerator: ProgramEnumerator[None], timeout: float = 60
    ) -> None:
        super()._init_task_solving_(task, enumerator, timeout)
        self._restarts = 0
        self._data: List[Tuple[Program, float]] = []
        self._last_size = 0

    def _close_task_solving_(
        self,
        task: Task[PBE],
        enumerator: ProgramEnumerator[None],
        time_used: float,
        solution: bool,
        last_program: Program,
    ) -> None:
        super()._close_task_solving_(
            task, enumerator, time_used, solution, last_program
        )
        self._stats["restarts"] += self._restarts

    def solve(
        self, task: Task[PBE], enumerator: ProgramEnumerator[None], timeout: float = 60
    ) -> Generator[Program, bool, None]:
        with chrono.clock(f"solve.{self.name()}.{self.subsolver.name()}") as c:  # type: ignore
            self._enumerator = enumerator
            self._init_task_solving_(task, self._enumerator, timeout)
            gen = self._enumerator.generator()
            program = next(gen)
            while program is not None:

                time = c.elapsed_time()
                if time >= timeout:
                    self._close_task_solving_(
                        task, self._enumerator, time, False, program
                    )
                    return
                self._programs += 1
                if self._test_(task, program):
                    should_stop = yield program
                    if should_stop:
                        self._close_task_solving_(
                            task, self._enumerator, time, True, program
                        )
                        return
                self._score = self.subsolver._score
                # Saves data
                if self._score > 0:
                    self._data.append((program, self._score))
                # If should restart
                if self._should_restart_():
                    self._restarts += 1
                    self._enumerator = self._restart_(self._enumerator)
                    gen = self._enumerator.generator()
                program = next(gen)

    def _should_restart_(self) -> bool:
        return self.restart_criterion(self)

    def _restart_(self, enumerator: ProgramEnumerator[None]) -> ProgramEnumerator[None]:
        pcfg = enumerator.G * 0  # type: ignore
        self._last_size = len(self._data)

        def reduce(
            score: float, S: Tuple[Type, Any], P: DerivableProgram, prob: float
        ) -> float:
            pcfg.probabilities[S][P] += score
            return score

        for program, score in self._data:
            pcfg.reduce_derivations(reduce, score, program)
        if self.uniform_prior > 0:
            pcfg = pcfg + (pcfg.uniform(pcfg.grammar) * self.uniform_prior)
        pcfg.normalise()
        new_enumerator = enumerator.clone(pcfg)
        return new_enumerator
