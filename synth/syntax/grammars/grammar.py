from abc import ABC, abstractmethod

from synth.syntax.program import Program


class Grammar(ABC):
    @abstractmethod
    def __contains__(self, program: Program) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of this class of grammar.
        """
        pass
