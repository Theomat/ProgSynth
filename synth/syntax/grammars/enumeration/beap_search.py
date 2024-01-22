from itertools import product
from heapq import heappush, heappop, heapify
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

from synth.filter.filter import Filter
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


class BeapSearch(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(
        self, G: ProbDetGrammar[U, V, W], filter: Optional[Filter[Program]] = None
    ) -> None:
        super().__init__(filter)
        assert isinstance(G.grammar, CFG)
        self.G = G
        self.cfg: CFG = G.grammar
        self._deleted: Set[Program] = set()

        # S -> cost list
        self._cost_lists: Dict[Tuple[Type, U], List[float]] = {}
        # S -> cost_index -> program list
        self._bank: Dict[Tuple[Type, U], Dict[int, List[Program]]] = {}
        # S -> heap of HeapElement queued
        self._queues: Dict[Tuple[Type, U], List[HeapElement]] = {}
        # S -> cost index set
        self._empties: Dict[Tuple[Type, U], Set[int]] = {}

        for S in self.G.grammar.rules:
            self._cost_lists[S] = []
            self._bank[S] = {}
            self._empties[S] = set()
            self._queues[S] = []

    def _init_non_terminal_(self, S: Tuple[Type, U]) -> None:
        if len(self._cost_lists[S]) > 0:
            return
        self._cost_lists[S].append(1e99)
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

    def _reevaluate_(self) -> None:
        if not self.cfg.is_recursive():
            return
        changed = True
        while changed:
            changed = False
            for S in list(self._queues.keys()):
                new_queue = [
                    HeapElement(
                        self.G.probabilities[S][el.P]
                        + sum(
                            self._cost_lists[self._non_terminal_for_(S, el.P, i)][0]
                            for i in range(len(el.combination))
                        ),
                        el.combination,
                        el.P,
                    )
                    for el in self._queues[S]
                ]
                if new_queue != self._queues[S]:
                    changed = True
                    heapify(new_queue)
                    self._queues[S] = new_queue
                    self._cost_lists[S][0] = self._queues[S][0].cost

    def generator(self) -> Generator[Program, None, None]:
        self._init_non_terminal_(self.G.start)
        self._reevaluate_()
        n = 0
        failed = False
        while not failed:
            self._failed_by_empties = False
            failed = True
            for prog in self.query(self.G.start, n):
                failed = False
                yield prog
            failed &= not self._failed_by_empties
            n += 1

    def programs_in_banks(self) -> int:
        return sum(sum(len(x) for x in val.values()) for val in self._bank.values())

    def programs_in_queues(self) -> int:
        return sum(len(val) for val in self._queues.values())

    def query(
        self, S: Tuple[Type, U], cost_index: int
    ) -> Generator[Program, None, None]:
        bank = self._bank[S]
        queue = self._queues[S]
        # When we return this way, it actually mean that we have generated all programs that this non terminal could generate
        if cost_index >= len(self._cost_lists[S]):
            return
        cost = self._cost_lists[S][cost_index]
        generated_program = False
        while queue and queue[0].cost == cost:
            element = heappop(queue)
            nargs = self.G.arguments_length_for(S, element.P)
            Sargs = [self._non_terminal_for_(S, element.P, i) for i in range(nargs)]
            # necessary for finite grammars
            failed = False
            failed_by_empties = False
            # Generate programs
            args_possibles = []
            for i in range(nargs):
                failed_by_empties, possibles = self._query_list_(
                    Sargs[i], element.combination[i]
                )
                if len(possibles) == 0:
                    failed = True
                    break
                args_possibles.append(possibles)
            if failed and not failed_by_empties:
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

            if failed_by_empties:
                continue

            if cost_index not in bank:
                bank[cost_index] = []
            for new_args in product(*args_possibles):
                if len(args_possibles) > 0:
                    new_program: Program = Function(element.P, list(new_args))
                else:
                    new_program = element.P
                if new_program in self._deleted:
                    continue
                elif not self._should_keep_subprogram(new_program):
                    self._deleted.add(new_program)
                    continue
                generated_program = True
                bank[cost_index].append(new_program)
                self._failed_by_empties = False
                yield new_program
        if not generated_program:
            self._empties[S].add(cost_index)
            self._failed_by_empties = True
        if queue:
            next_cost = queue[0].cost
            self._cost_lists[S].append(next_cost)

    def _query_list_(
        self, S: Tuple[Type, U], cost_index: int
    ) -> Tuple[bool, List[Program]]:
        """
        returns failed_by_empties, programs
        """
        # It's an empty cost index
        if cost_index in self._empties[S]:
            return True, []
        bank = self._bank[S]
        if cost_index in bank:
            return False, bank[cost_index]
        for x in self.query(S, cost_index):
            pass
        return False, bank[cost_index]

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
        return "beap-search"

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "BeapSearch[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        enum._deleted = self._deleted.copy()
        return enum


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W]) -> BeapSearch[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: -np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return BeapSearch(Gp)
