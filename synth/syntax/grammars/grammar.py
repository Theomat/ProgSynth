from abc import ABC, abstractmethod

from synth.syntax.program import Program


class Grammar(ABC):
    @abstractmethod
    def __contains__(self, program: Program) -> bool:
        pass
