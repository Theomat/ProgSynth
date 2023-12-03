from collections import defaultdict
from itertools import product
from heapq import heappush, heappop
from typing import (
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


class BeeSearch(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(self, G: ProbDetGrammar[U, V, W]) -> None:
        assert isinstance(G.grammar, CFG)
        self.G = G
        self._seen: Set[Program] = set()
        self._deleted: Set[Program] = set()

        self._cost_list: List[float] = []
        # S -> cost_index -> program list
        self._bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = {}
        # S -> heap of HeapElement queued
        self._prog_queued: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> max index currently queued
        self._max_index: Dict[Tuple[Type, U], int] = defaultdict(int)
        self._delayed: Dict[
            Tuple[Type, U], List[Tuple[List[int], DerivableProgram, Optional[int]]]
        ] = defaultdict(list)
        # Fill terminals first
        for S in self.G.rules:
            self._bank[S] = {}
            self._prog_queued[S] = []

            for P in self.G.rules[S]:
                nargs = self.G.arguments_length_for(S, P)
                if nargs == 0:
                    self._add_combination_(S, P, [])

        # Init non terminals (otherwise add combination won't work correctly)
        for S in self.G.rules:
            for P in self.G.rules[S]:
                nargs = self.G.arguments_length_for(S, P)
                if nargs > 0:
                    index_cost = [0] * nargs
                    self._add_combination_(S, P, index_cost)

    def _add_combination_(
        self,
        S: Tuple[Type, U],
        P: DerivableProgram,
        index_cost: List[int],
        changed_index: Optional[int] = None,
    ) -> None:
        # Check if it needs to be delayed or not
        to_check = (
            [changed_index]
            if changed_index is not None
            else [i for i in range(len(index_cost))]
        )
        for i in to_check:
            if index_cost[i] >= len(self._cost_list):
                self._delayed[S].append((index_cost, P, changed_index))
                return
        # No need to delay add it
        new_cost = self._index_cost2real_cost_(S, P, index_cost)
        heappush(
            self._prog_queued[S],
            HeapElement(new_cost, index_cost, P),
        )

    def _trigger_delayed_(self) -> None:
        copy = self._delayed.copy()
        self._delayed.clear()
        for S, elements in copy.items():
            for index_cost, P, to_check in elements:
                self._add_combination_(S, P, index_cost, to_check)

    def _add_cost_(self, S: Tuple[Type, U], cost: float) -> Tuple[bool, int]:
        cost_list = self._cost_list
        if len(cost_list) > 0 and cost_list[-1] == cost:
            return False, len(cost_list) - 1
        # assert len(cost_list) == 0 or cost > cost_list[-1], f"{cost} -> {cost_list}"
        # print("adding:", cost, "to", cost_list)
        cost_list.append(cost)
        self._trigger_delayed_()
        return True, len(cost_list) - 1

    def _add_program_(
        self, S: Tuple[Type, U], new_program: Program, cost_index: int
    ) -> None:
        if new_program in self._deleted:
            return
        local_bank = self._bank[S]
        if cost_index not in local_bank:
            local_bank[cost_index] = []
        # assert max(local_bank.keys()) + 1 == len(self._cost_lists[S]), f"index:{cost_index} {local_bank.keys()} vs {len(self._cost_lists[S])}"
        local_bank[cost_index].append(new_program)

    def _index_cost2real_cost_(
        self, S: Tuple[Type, U], P: DerivableProgram, indices: List[int]
    ) -> float:
        out = self.G.probabilities[S][P]
        for i in range(self.G.arguments_length_for(S, P)):
            out += self._cost_list[indices[i]]
        return out

    def _non_terminal_for_(
        self, S: Tuple[Type, U], P: DerivableProgram, index: int
    ) -> Tuple[Type, U]:
        Sp = self.G.rules[S][P][0][index]  # type: ignore
        return (Sp[0], (Sp[1], None))  # type: ignore

    def generator(self) -> Generator[Program, None, None]:
        progs = self.G.programs()
        while True:
            non_terminals, cost = self._next_cheapest_()
            if cost is None:
                break
            if len(non_terminals) == 0:
                break
            for program in self._produce_programs_from_cost_(non_terminals, cost):
                progs -= 1
                if program in self._seen:
                    continue
                self._seen.add(program)
                yield program
            if progs <= 0:
                break

    def _next_cheapest_(self) -> Tuple[List[Tuple[Type, U]], Optional[float]]:
        """
        WORKS
        """
        cheapest = None
        non_terminals_container: List[Tuple[Type, U]] = []
        for S, heap in self._prog_queued.items():
            if len(heap) == 0:
                continue
            item = heap[0]
            smallest_cost = item.cost
            if cheapest is None or smallest_cost <= cheapest:
                if cheapest is None or smallest_cost < cheapest:
                    non_terminals_container = []
                    cheapest = smallest_cost
                non_terminals_container.append(S)
        return non_terminals_container, cheapest

    def _produce_programs_from_cost_(
        self, non_terminals: List[Tuple[Type, U]], cost: float
    ) -> Generator[Program, None, None]:
        for S in non_terminals:
            queue = self._prog_queued[S]
            maxi = self._max_index[S]
            # Add cost since we are pre-generative
            cost_index = self._add_cost_(S, cost)[1]

            while queue and queue[0].cost == cost:
                element = heappop(queue)
                assert element.cost == cost
                nargs = self.G.arguments_length_for(S, element.P)
                Sargs = [self._non_terminal_for_(S, element.P, i) for i in range(nargs)]
                # Generate next combinations
                for i in range(nargs):
                    index_cost = element.combination.copy()
                    index_cost[i] += 1
                    self._add_combination_(S, element.P, index_cost, i)
                    if index_cost[i] > maxi:
                        maxi = index_cost[i]
                    # Avoid duplication with this condition
                    if index_cost[i] > 1:
                        break
                # Generate programs
                args_possibles = []
                for i in range(nargs):
                    local_bank = self._bank[Sargs[i]]
                    ci = element.combination[i]
                    if ci not in local_bank or len(local_bank[ci]) == 0:
                        break
                    args_possibles.append(local_bank[ci])
                if len(args_possibles) != nargs:
                    # print("failed")
                    continue
                for new_args in product(*args_possibles):
                    new_program = Function(element.P, list(new_args))
                    self._add_program_(S, new_program, cost_index)
                    if S == self.G.start:
                        yield new_program
            self._max_index[S] = maxi

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
        return "bee-search"

    def clone_with_memory(
        self, G: Union[ProbDetGrammar, ProbUGrammar]
    ) -> "BeeSearch[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        enum._seen = self._seen.copy()
        return enum


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W]) -> BeeSearch[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: -np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return BeeSearch(Gp)
