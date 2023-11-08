from typing import (
    Generator,
    Generic,
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
    def probability(self, program: Program) -> float:
        """
        Return the probability of generating the given program according to the grammar associated with this enumerator.
        """
        pass

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        Function used for observational equivalence, that means other and representative are semantically equivalent for the current task.
        """
        pass

    @abstractmethod
    def clone_with_memory(
        self, grammar: Union[ProbDetGrammar, ProbUGrammar]
    ) -> "ProgramEnumerator[U]":
        """
        Clone this enumerator with the specified new grammar but remember every single program enumerated so that it does not enumerate them again.
        """
        pass
