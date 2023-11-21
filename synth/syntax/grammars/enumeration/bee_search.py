from collections import defaultdict
from itertools import product
from heapq import heappush, heappop, heapify
from typing import (
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass
from abc import ABC
import bisect

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
    order: int
    combination: List[int]
    P: DerivableProgram

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combination}, {self.P})"


class BeeSearch(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(self, G: ProbDetGrammar[U, V, W]) -> None:
        assert isinstance(G.grammar, CFG)
        self.G: ProbDetGrammar = ProbDetGrammar(
            G.grammar,
            {
                S: {P: np.log(p) for P, p in val.items() if p > 0}
                for S, val in G.probabilities.items()
            },
        )
        self._seen: Set[Program] = set()

        # S -> cost list
        self._cost_lists: Dict[Tuple[Type, U], List[float]] = {}
        # S -> cost_index -> program list
        self._bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = {}
        # S -> heap of HeapElement queued
        self._prog_queued: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> heap of HeapElement popped
        self._prog_popped: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> max index used
        self._max_index: Dict[Tuple[Type, U], int] = defaultdict(int)
        self._delayed: Dict[
            Tuple[Type, U],
            Dict[
                Tuple[Type, U], List[Tuple[List[int], DerivableProgram, Optional[int]]]
            ],
        ] = {}
        # S -> set of S' such that S' may consume a S as argument
        self._consumers_of: Dict[Tuple[Type, U], Set[Tuple[Type, U]]] = defaultdict(set)
        self._heapify_queue: Set[Tuple[Type, U]] = set()

        # To break equality (is this really needed? perhaps they do not know compare=False?)
        self.order = 0

        self._terminals: List[HeapElement] = []

        # Fill terminals first
        for S in self.G.rules:
            self._bank[S] = {}
            self._cost_lists[S] = []
            self._prog_popped[S] = []
            self._prog_queued[S] = []
            self._delayed[S] = defaultdict(list)

            for P in self.G.rules[S]:
                nargs = self.G.arguments_length_for(S, P)
                if nargs == 0:
                    new_priority = self._index_cost2real_cost_(S, P, [])
                    self._add_program_(S, P, new_priority)
                    if S == self.G.start:
                        self.order += 1
                        heappush(
                            self._terminals,
                            HeapElement(new_priority, self.order, [], P),
                        )

        # Init non terminals
        for S in self.G.rules:
            for P in self.G.rules[S]:
                nargs = self.G.arguments_length_for(S, P)
                if nargs > 0:
                    index_cost = [0] * nargs
                    self._add_combination_(S, P, index_cost)
                    for i in range(nargs):
                        self._consumers_of[self._non_terminal_for_(S, P, i)].add(S)

    def _add_combination_(
        self,
        S: Tuple[Type, U],
        P: DerivableProgram,
        index_cost: List[int],
        changed_index: Optional[int] = None,
    ) -> None:
        # Check if it needs to be delayed or not
        if changed_index is not None:
            i = changed_index
            Sp = self._non_terminal_for_(S, P, i)
            if index_cost[i] >= len(self._cost_lists[Sp]):
                self._delayed[Sp][S].append((index_cost, P, changed_index))
                return
            pass
        else:
            for i in range(len(index_cost)):
                Sp = self._non_terminal_for_(S, P, i)
                if index_cost[i] >= len(self._cost_lists[Sp]):
                    self._delayed[Sp][S].append((index_cost, P, None))
                    return
        # No need to delay add it
        # print(
        #     "QUEUED:",
        #     S,
        #     "->",
        #     P,
        #     ":",
        #     index_cost,
        #     "[",
        #     [
        #         max(self._bank[self._non_terminal_for_(S, P, i)].keys())
        #         for i in range(len(index_cost))
        #     ],
        #     "]",
        # )
        new_cost = self._index_cost2real_cost_(S, P, index_cost)
        heappush(
            self._prog_queued[S],
            HeapElement(new_cost, self.order, index_cost, P),
        )
        self.order += 1

    def _heapify_(self) -> None:
        """
        We need to re compute the cost of our current tuples for some S since we updated our cost list.
        """
        for S in self._heapify_queue:
            queue = self._prog_queued[S]
            new_queue = [
                HeapElement(
                    self._index_cost2real_cost_(S, e.P, e.combination),
                    e.order,
                    e.combination,
                    e.P,
                )
                for e in queue
            ]
            self._prog_queued[S] = new_queue
            heapify(new_queue)
        self._heapify_queue.clear()

    def _add_program_(
        self, S: Tuple[Type, U], new_program: Program, priority: float
    ) -> None:

        cost_list = self._cost_lists[S]
        cost_index = bisect.bisect(cost_list, priority)
        # There's a shift by one to do depending if priority exist or not in cost_list
        # If cost does not exist add it
        if cost_index - 1 < 0 or cost_list[cost_index - 1] != priority:
            if cost_index < self._max_index[S]:
                self._heapify_queue |= self._consumers_of[S]
                print("COST SHOULD NOT HAPPEN")
            cost_list.insert(cost_index, priority)

            # Trigger delayed
            delayed = self._delayed[S]
            copy = delayed.copy()
            delayed.clear()
            for Sp, elements in copy.items():
                for index_cost, P, changed_index in elements:
                    self._add_combination_(Sp, P, index_cost, changed_index)
        else:
            cost_index -= 1
        # Add it to the bank also
        local_bank = self._bank[S]
        # print("ADD:", S, "at cost=", cost_index, cost_list)
        if cost_index not in local_bank:
            local_bank[cost_index] = []
            # print("INIT from", S, ":", priority, "in", cost_list)
        local_bank[cost_index].append(new_program)

    def _index_cost2real_cost_(
        self, S: Tuple[Type, U], P: DerivableProgram, indices: List[int]
    ) -> float:
        out = -self.G.probabilities[S][P]
        for i in range(self.G.arguments_length_for(S, P)):
            out += self._cost_lists[self._non_terminal_for_(S, P, i)][indices[i]]
        return out

    def _non_terminal_for_(
        self, S: Tuple[Type, U], P: DerivableProgram, index: int
    ) -> Tuple[Type, U]:
        Sp = self.G.rules[S][P][0][index]
        return (Sp[0], (Sp[1], None))  # type: ignore

    def generator(self) -> Generator[Program, None, None]:
        while True:
            non_terminals, cost = self._next_cheapest_()
            if cost is None:
                break
            # print("cost=", cost)
            if len(non_terminals) == 0:
                if self._terminals:
                    program: Program = heappop(self._terminals).P
                    if program in self._seen:
                        continue
                    self._seen.add(program)
                    # print("\t>", program)

                    yield program
                    continue
                else:
                    break
            for program in self._produce_programs_from_cost_(non_terminals, cost):
                if program in self._seen:
                    continue
                self._seen.add(program)
                # print("\t>", program)
                yield program
            self._heapify_()
            self._generate_next_combinations_(non_terminals)

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
        if len(self._terminals) > 0 and (
            cheapest is None or self._terminals[0].cost <= cheapest
        ):
            cheapest = self._terminals[0].cost
            non_terminals_container = []
        return non_terminals_container, cheapest

    def _generate_next_combinations_(self, non_terminals: List[Tuple[Type, U]]) -> None:
        for S in non_terminals:
            maxi = self._max_index[S]
            popped = self._prog_popped[S]
            for element in popped:
                print(S, "from", element.combination)
                for i in range(len(element.combination)):
                    index_cost = element.combination.copy()
                    l = len(self._cost_lists[self._non_terminal_for_(S, element.P, i)])
                    if index_cost[i] == l - 1:
                        print("index_cost:", index_cost, "i=", i, "len=", l)
                        print("WHY THEY DID NOT GET GENERATED")
                        break
                    index_cost[i] += 1
                    self._add_combination_(S, element.P, index_cost, i)
                    if index_cost[i] > maxi:
                        maxi = index_cost[i]
                    if index_cost[i] > 1:
                        break
            self._max_index[S] = maxi
            popped.clear()

    def _produce_programs_from_cost_(
        self, non_terminals: List[Tuple[Type, U]], cost: float
    ) -> Generator[Program, None, None]:
        """SHOULD WORK"""
        for S in non_terminals:
            queue = self._prog_queued[S]
            popped = self._prog_popped[S]
            while queue and queue[0].cost == cost:
                element = heappop(queue)
                assert element.cost == cost
                popped.append(element)

                print(f"\t{S} => {element.P} combination:", element.combination)

                args_possibles = []
                nargs = self.G.arguments_length_for(S, element.P)
                for i in range(nargs):
                    Sp = self._non_terminal_for_(S, element.P, i)
                    local_bank = self._bank[Sp]
                    # print(f"\t\targ[{i}] => {Sp} bank keys:", local_bank.keys())
                    args_possibles.append(local_bank[element.combination[i]])
                    # print("\t\t\tpossibles=", local_bank[element.combination[i]])
                for new_args in product(*args_possibles):
                    new_program = Function(element.P, list(new_args))
                    self._add_program_(S, new_program, element.cost)

                    # prob = self.G.probability(new_program, start=S)
                    # print("\tprob=", prob, "element.cost=", element.cost, "call:", self._index_cost2real_cost_(S, element.P, element.combination))
                    # assert -prob == element.cost
                    if S == self.G.start:
                        yield new_program

    def merge_program(self, representative: Program, other: Program) -> None:
        pass

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
    return BeeSearch(G)
