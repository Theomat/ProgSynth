from heapq import heappush, heappop
from typing import (
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field

import numpy as np

from synth.filter.filter import Filter
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.program import Function, Program
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar, DerivableProgram
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


def _build_(
    elems: List[Tuple[DerivableProgram, Tuple[Type, U]]], G: ProbDetGrammar[U, V, W]
) -> Program:
    P, S = elems.pop(0)
    nargs = G.arguments_length_for(S, P)
    if nargs == 0:
        return P
    else:
        args = []
        while nargs > 0:
            args.append(_build_(elems, G))
            nargs -= 1
        return Function(P, args)


@dataclass(order=True, frozen=True)
class HeapElement(Generic[U]):
    priority: float
    to_expand: List[Tuple[Type, U]] = field(compare=False)
    parts: List[Tuple[DerivableProgram, Tuple[Type, U]]] = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.priority}, {self.parts})"

    def make_program(self, g: ProbDetGrammar[U, V, W]) -> Program:
        return _build_(self.parts, g)


class AStar(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(
        self,
        G: ProbDetGrammar[U, V, W],
        filter: Optional[Filter[Program]] = None,
    ) -> None:
        super().__init__(filter)
        self.current: Optional[Program] = None

        self.G = G
        self.start = G.start
        self.rules = G.rules

        self.frontier: List[HeapElement[U]] = []

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "a-star"

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        first = (self.G.start[0], self.G.start[1][0])  # type: ignore
        heappush(self.frontier, HeapElement(0, [first], []))

        while self.frontier:
            elem = heappop(self.frontier)
            if len(elem.to_expand) == 0:
                p = elem.make_program(self.G)
                if self._should_keep_subprogram(p):
                    yield p
            else:
                partS = elem.to_expand.pop()
                S = (partS[0], (partS[1], None))
                for P in self.G.rules[S]:  # type: ignore
                    args = self.G.rules[S][P][0]  # type: ignore
                    p = self.G.probabilities[S][P]  # type: ignore
                    new_el = HeapElement(
                        elem.priority + p,  # type: ignore
                        elem.to_expand + list(args),
                        elem.parts + [(P, S)],
                    )
                    heappush(self.frontier, new_el)

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        In other words, other will no longer be generated through heap search
        """
        pass

    def programs_in_banks(self) -> int:
        return 0

    def programs_in_queues(self) -> int:
        return len(self.frontier)

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "AStar[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        return enum


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W]) -> AStar[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: -np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return AStar(Gp)
