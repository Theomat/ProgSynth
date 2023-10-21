from typing import (
    Generator,
    Generic,
    TypeVar,
)
from abc import ABC, abstractmethod

from synth.syntax.program import Program


U = TypeVar("U")


class ProgramEnumerator(ABC, Generic[U]):
    """
    Object that enumerates over programs.
    When a program is generated a feedback of type U is expected.
    If U is None then no feedback is expected.ss
    """

    @abstractmethod
    @classmethod
    def name(cls) -> str:
        pass

    @abstractmethod
    def generator(self) -> Generator[Program, U, None]:
        pass

    def __iter__(self) -> Generator[Program, U, None]:
        return self.generator()

    @abstractmethod
    def probability(self, program: Program) -> float:
        pass

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        Function used for observational equivalence, that means other and representative are semantically equivalent for the current task.
        """
        pass
