from collections import defaultdict
from heapq import heappush, heappop
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class HeapElement:
    cost_tuple: Tuple[int, ...]
    context: Tuple[Tuple[Type, Any], DerivableProgram] = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.context[0]} -> {self.context[1]}: {self.cost_tuple})"


class BSEnumerator(
    ProgramEnumerator[None],
    ABC,
    Generic[U, V, W],
):
    def __init__(
        self, G: ProbDetGrammar[U, V, W]) -> None:
        self.G = G
        self.bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.cost_map: Dict[int, int] = {}
        self.Q: List[HeapElement] = []
        self._seen: Dict[
            Tuple[Type, U], Dict[DerivableProgram, Set[Tuple[int, ...]]]
        ] = {}

    def __iter__(self) -> Generator[Program, None, None]:
        self._init_heap_()
        to_add: List[Tuple[Tuple[Type, U], int, Program]] = []
        while True:
            # Add programs to bank
            for S, cost, program in to_add:
                self.bank[S][cost].append(program)
            to_add.clear()
            # Generate next batch of programs
            elem = self.query()
            (S, P) = elem.context
            cost_tuple = elem.cost_tuple
            for program, cost in self._programs_from_(S, P, cost_tuple):
                to_add.append((S, cost, program))
                yield program

    def _init_heap_(self) -> None:
        max_priority = 1
        for S in self.G.rules:
            self._seen[S] = {}
            for P in self.G.rules[S]:
                self._seen[S][P] = set()
                nargs = self.G.arguments_length_for(S, P)
                cost_tuple = tuple(1 for _ in range(nargs))
                heappush(self.Q, HeapElement(cost_tuple, (S, P)))
                self._seen[S][P].add(cost_tuple)
                # Init bank with terminals
                if nargs == 0:
                    self.bank[S][1].append(P)
                    max_priority = min(max_priority, self.compute_priority(P, []))

    def _possible_args_(
        self, info: W, S: Tuple[Type, U], cost_tuple: Tuple[int, ...], n: int
    ) -> Generator[List[Program], None, None]:
        if n >= len(cost_tuple):
            yield []
            return
        for program in self.bank[S][self.cost_map[cost_tuple[n]]]:
            new_info, Slist = self.G.derive_all(info, S, program)
            for continuation in self._possible_args_(
                new_info, Slist[-1], cost_tuple, n + 1
            ):
                yield [program] + continuation

    def _programs_from_(
        self, S: Tuple[Type, U], P: DerivableProgram, cost_tuple: Tuple[int, ...]
    ) -> Generator[Tuple[Program, int], None, None]:
        info, S = self.G.derive(self.G.start_information(), S, P)
        for args in self._possible_args_(info, S, cost_tuple, 0):
            yield Function(P, args), self.compute_priority(P, args)

    def query(self) -> HeapElement:
        elem = heappop(self.Q)
        k = len(elem.cost_tuple)
        # C = C U w(n)
        if k == 0:
            return elem
        (S, P) = elem.context
        for i in range(k):
            new_cost_tuple = tuple(
                k + (ik == i) for ik, k in enumerate(elem.cost_tuple)
            )
            if new_cost_tuple not in self._seen[S][P]:
                n_prime = HeapElement(new_cost_tuple, elem.context)
                heappush(self.Q, n_prime)
                self._seen[S][P].add(new_cost_tuple)
        return elem

    @classmethod
    def name(cls) -> str:
        return "bee-search"

    @abstractmethod
    def compute_priority(self, P: DerivableProgram, args: List[Program]) -> Ordered:
        pass
