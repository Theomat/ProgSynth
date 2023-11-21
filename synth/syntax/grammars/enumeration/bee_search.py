from itertools import product
from heapq import heappush, heappop
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
from abc import ABC, abstractmethod
import bisect
from synth.syntax.grammars.cfg import CFG

from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class HeapElement:
    cost: Ordered
    order: int
    combination: List[int]
    P: DerivableProgram

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combination}, {self.P})"


class BSEnumerator(
    ProgramEnumerator[None],
    ABC,
    Generic[U, V, W],
):
    def __init__(self, G: ProbDetGrammar[U, V, W]) -> None:
        assert isinstance(G.grammar, CFG)
        self.G = G
        self._seen: Set[Program] = set()

        # S -> cost list
        self._cost_lists: Dict[Tuple[Type, U], List[Ordered]] = {}
        # S -> cost_index -> program list
        self._bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = {}
        # S -> heap of HeapElement queued
        self._prog_queued: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> heap of HeapElement popped
        self._prog_popped: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> max index used
        self._prog_last_max: Dict[Tuple[Type, U], int] = {}

        # To break equality (is this really needed? perhaps they do not know compare=False?)
        self.order = 0

        self._terminals: List[HeapElement] = []

        # Fill terminals first
        for S in self.G.rules:
            self._bank[S] = {}
            self._cost_lists[S] = []
            self._prog_popped[S] = []
            self._prog_queued[S] = []
            self._prog_last_max[S] = 0

            for P in self.G.rules[S]:
                nargs = self.G.arguments_length_for(S, P)
                if nargs == 0:
                    new_priority = self.compute_priority(S, P, [])
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
                    cost = self._index_cost2real_cost_(S, P, index_cost)
                    self.order += 1
                    heappush(
                        self._prog_queued[S],
                        HeapElement(cost, self.order, index_cost, P),
                    )

    def _add_program_(
        self, S: Tuple[Type, U], new_program: Program, priority: Ordered
    ) -> None:

        cost_list = self._cost_lists[S]
        cost_index = bisect.bisect(cost_list, priority)
        # If cost does not exist add it
        if cost_index >= len(cost_list) or cost_list[cost_index] != priority:
            bisect.insort(
                cost_list,
                priority,
                lo=max(0, cost_index - 1),
                hi=min(cost_index + 1, len(cost_list)),
            )
        # Add it to the bank also
        local_bank = self._bank[S]
        if cost_index not in local_bank:
            local_bank[cost_index] = []
        local_bank[cost_index].append(new_program)

    def _index_cost2real_cost_(
        self, S: Tuple[Type, U], P: DerivableProgram, indices: List[int]
    ) -> Ordered:
        out = []
        for i in range(self.G.arguments_length_for(S, P)):
            out.append(self._cost_lists[self._non_terminal_for_(S, P, i)][indices[i]])
        return self.compute_priority(S, P, out)

    def _non_terminal_for_(
        self, S: Tuple[Type, U], P: DerivableProgram, index: int
    ) -> Tuple[Type, U]:
        Sp = self.G.rules[S][P][0][index]  # type: ignore
        # print("Fom S=", S, "getting:", Sp, "->", (Sp[0], (Sp[1], None)))
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
            for program in self._expand_from_(non_terminals, cost):
                if program in self._seen:
                    continue
                self._seen.add(program)
                # print("\t>", program)
                yield program
            self._next_combination_()

    def _next_cheapest_(self) -> Tuple[List[Tuple[Type, U]], Optional[Ordered]]:
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

    def _next_combination_(self) -> None:
        for S in self._prog_queued:
            popped = self._prog_popped[S]
            if len(popped) == 0:
                continue
            queue = self._prog_queued[S]
            max_index_used = self._prog_last_max[S]
            for element in popped:
                # print("from", element.combination)
                for i in range(len(element.combination)):
                    index_cost = element.combination.copy()
                    if index_cost[i] + 1 >= len(
                        self._cost_lists[self._non_terminal_for_(S, element.P, i)]
                    ):
                        continue
                    index_cost[i] += 1
                    # print("\t->", index_cost)
                    if index_cost[i] > max_index_used:
                        max_index_used = index_cost[i]
                    new_cost = self._index_cost2real_cost_(S, element.P, index_cost)
                    heappush(
                        queue,
                        HeapElement(new_cost, self.order, index_cost, element.P),
                    )
                    self.order += 1
                    if index_cost[i] > 1:
                        break
            popped.clear()
            self._prog_last_max[S] = max_index_used

    def _expand_from_(
        self, non_terminals: List[Tuple[Type, U]], cost: Ordered
    ) -> Generator[Program, None, None]:
        for S in non_terminals:
            queue = self._prog_queued[S]
            popped = self._prog_popped[S]
            while queue and queue[0].cost == cost:
                element = heappop(queue)
                assert element.cost == cost
                popped.append(element)

                # print("\tcombination:", element.combination)

                args_possibles = []
                info, Sp = self.G.derive(self.G.start_information(), S, element.P)
                nargs = self.G.arguments_length_for(S, element.P)
                failed = False
                for i in range(nargs):
                    local_bank = self._bank[Sp]
                    # print(f"\t\targ[{i}] => {Sp}")
                    args_possibles.append(local_bank[element.combination[i]])
                    first = local_bank[element.combination[i]][0]
                    # print("\t\t\tpossibles=", local_bank[element.combination[i]])
                    # print("\t\t\tfirst=", first)
                    if i + 1 < nargs:
                        info, S2 = self.G.derive_all(info, Sp, first)
                        Sp = S2[-1]
                if failed:
                    continue
                for new_args in product(*args_possibles):

                    new_program = Function(element.P, list(new_args))
                    self._add_program_(S, new_program, element.cost)

                    prob = self.G.probability(new_program, start=S)
                    # print("\tprob=", prob, "element.cost=", element.cost)
                    assert -prob == element.cost
                    yield new_program

    def merge_program(self, representative: Program, other: Program) -> None:
        pass

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "bee-search"

    @abstractmethod
    def compute_priority(
        self, S: Tuple[Type, U], P: DerivableProgram, args_cost: Iterable[Ordered]
    ) -> Ordered:
        pass

    def clone_with_memory(
        self, G: Union[ProbDetGrammar, ProbUGrammar]
    ) -> "BSEnumerator[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        enum._seen = self._seen.copy()
        return enum


class BeeSearch(BSEnumerator[U, V, W]):
    def compute_priority(
        self, S: Tuple[Type, U], P: DerivableProgram, args_cost: Iterable[Ordered]
    ) -> Ordered:
        p = self.G.probabilities[S][P]
        for c in args_cost:
            p *= c  # type: ignore
        return -p


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W]) -> BeeSearch[U, V, W]:
    return BeeSearch(G)
