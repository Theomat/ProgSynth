from collections import defaultdict
from itertools import product
from heapq import heappush, heappop
from typing import (
    Dict,
    Generator,
    Generic,
    List,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field

import numpy as np

from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class HeapElement:
    cost: float
    combination: List[int]
    P: DerivableProgram = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combination}, {self.P})"


class BeepSearch(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(self, G: ProbDetGrammar[U, V, W]) -> None:
        assert isinstance(G.grammar, CFG)
        self.G = G
        self._seen: Set[Program] = set()
        self._deleted: Set[Program] = set()

        # S -> cost list
        self._cost_lists: Dict[Tuple[Type, U], List[float]] = defaultdict(list)
        # S -> cost_index -> program list
        self._bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = defaultdict(dict)
        # S -> heap of HeapElement queued
        self._queues: Dict[Tuple[Type, U], List[HeapElement]] = defaultdict(list)

    def _init_non_terminal_(self, S: Tuple[Type, U]) -> None:
        if len(self._cost_lists[S]) > 0:
            return
        self._cost_lists[S].append(0)
        queue = self._queues[S]
        for P in self.G.rules[S]:
            # Init args
            nargs = self.G.arguments_length_for(S, P)
            cost = self.G.probabilities[S][P]
            for i in range(nargs):
                Si = self._non_terminal_for_(S, P, i)
                self._init_non_terminal_(Si)
                cost += self._cost_lists[Si][0]
            index_cost = [0] * nargs
            heappush(queue, HeapElement(cost, index_cost, P))

        self._cost_lists[S][0] = queue[0].cost

    def _non_terminal_for_(
        self, S: Tuple[Type, U], P: DerivableProgram, index: int
    ) -> Tuple[Type, U]:
        Sp = self.G.rules[S][P][0][index]  # type: ignore
        return (Sp[0], (Sp[1], None))  # type: ignore

    def generator(self) -> Generator[Program, None, None]:
        self._init_non_terminal_(self.G.start)
        n = 0
        failed = False
        while not failed:
            failed = True
            for prog in self.query(self.G.start, n):
                failed = False
                if prog in self._seen:
                    continue
                self._seen.add(prog)
                yield prog
            n += 1

    def query(
        self, S: Tuple[Type, U], cost_index: int
    ) -> Generator[Program, None, None]:
        bank = self._bank[S]
        if cost_index in bank:
            for prog in bank[cost_index]:
                yield prog
            return

        queue = self._queues[S]
        if cost_index >= len(self._cost_lists[S]):
            return
        cost = self._cost_lists[S][cost_index]
        while queue and queue[0].cost == cost:
            element = heappop(queue)
            nargs = self.G.arguments_length_for(S, element.P)
            Sargs = [self._non_terminal_for_(S, element.P, i) for i in range(nargs)]
            failed = False
            # Generate programs
            args_possibles = []
            for i in range(nargs):
                possibles = list(self.query(Sargs[i], element.combination[i]))
                if len(possibles) == 0:
                    failed = True
                    break
                args_possibles.append(possibles)
            if failed:
                continue
            # Generate next combinations
            for i in range(nargs):
                index_cost = element.combination.copy()
                index_cost[i] += 1
                cl = self._cost_lists[Sargs[i]]
                if index_cost[i] >= len(cl):
                    if index_cost[i] > 1:
                        break
                    continue
                new_cost = cost - cl[index_cost[i] - 1] + cl[index_cost[i]]
                heappush(queue, HeapElement(new_cost, index_cost, element.P))
                # Avoid duplication with this condition
                if index_cost[i] > 1:
                    break

            if cost_index not in bank:
                bank[cost_index] = []
            for new_args in product(*args_possibles):
                new_program = Function(element.P, list(new_args))
                if new_program in self._deleted:
                    continue
                bank[cost_index].append(new_program)
                yield new_program

        if queue:
            next_cost = queue[0].cost
            self._cost_lists[S].append(next_cost)

    def merge_program(self, representative: Program, other: Program) -> None:
        self._deleted.add(other)
        for S in self.G.rules:
            if S[0] != other.type:
                continue
            local_bank = self._bank[S]
            for programs in local_bank.values():
                if other in programs:
                    programs.remove(other)

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "beep-search"

    def clone_with_memory(
        self, G: Union[ProbDetGrammar, ProbUGrammar]
    ) -> "BeepSearch[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        enum._seen = self._seen.copy()
        enum._deleted = self._deleted.copy()
        return enum


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W]) -> BeepSearch[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: -np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return BeepSearch(Gp)
