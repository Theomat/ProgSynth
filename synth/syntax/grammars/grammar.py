from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from synth.syntax.program import Constant, Primitive, Program, Variable

DerivableProgram = Union[Primitive, Variable, Constant]


@dataclass(frozen=True)
class NGram:
    n: int
    predecessors: List[Tuple[DerivableProgram, int]] = field(default_factory=lambda: [])

    def __hash__(self) -> int:
        return hash((self.n, tuple(self.predecessors)))

    def __str__(self) -> str:
        return str(self.predecessors)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.predecessors)

    def successor(self, new_succ: Tuple[DerivableProgram, int]) -> "NGram":
        new_pred = [new_succ] + self.predecessors
        if len(new_pred) + 1 > self.n and self.n >= 0:
            new_pred.pop()
        return NGram(self.n, new_pred)

    def last(self) -> Tuple[DerivableProgram, int]:
        return self.predecessors[0]


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

    @abstractmethod
    def clean(self) -> None:
        """
        Clean the grammar.
        """
        pass
