from typing import (
    Callable,
    Generator,
    Generic,
    Optional,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar

from synth.syntax.program import Program


U = TypeVar("U")


class ProgramEnumerator(ABC, Generic[U]):
    """
    Object that enumerates over programs.
    When a program is generated a feedback of type U is expected.
    If U is None then no feedback is expected.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @abstractmethod
    def generator(self) -> Generator[Program, U, None]:
        pass

    def __iter__(self) -> Generator[Program, U, None]:
        return self.generator()

    @abstractmethod
    def programs_in_banks(self) -> int:
        pass

    @abstractmethod
    def programs_in_queues(self) -> int:
        pass

    @abstractmethod
    def probability(self, program: Program) -> float:
        """
        Return the probability of generating the given program according to the grammar associated with this enumerator.
        """
        pass

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        Function used for observational equivalence, that means other and representative are semantically equivalent for the current task.
        This is for a posteriori merging, it is rather inefficient compared to evaluating subprograms for most enumerative algorithms.
        """
        pass

    def set_subprogram_filter(self, filter: Callable[[Program], bool]) -> None:
        """
        Function that can be called to filter out subprograms, can be used for observational equivalence and is the most efficient way to do so.
        When filter returns True the program will be discarded by the enumerator.
        """
        self._filter: Optional[Callable[[Program], bool]] = filter

    def _should_keep_subprogram(self, program: Program) -> bool:
        return self._filter is None or not self._filter(program)

    @abstractmethod
    def clone_with_memory(
        self, grammar: Union[ProbDetGrammar, ProbUGrammar]
    ) -> "ProgramEnumerator[U]":
        """
        Clone this enumerator with the specified new grammar but remember every single program enumerated so that it does not enumerate them again.
        """
        pass
