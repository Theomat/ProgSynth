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
from synth.syntax.grammars.enumeration.constant_delay_queue import CDQueue, CostTuple
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")



@dataclass(order=True, frozen=True)
class Derivation:
    cost: float
    combination: int
    P: DerivableProgram = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combination}, {self.P})"



class CDSearch(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(
        self, G: ProbDetGrammar[U, V, W], filter: Optional[Filter[Program]] = None, k: int = 5, precision: float = 1e-5
    ) -> None:
        super().__init__(filter)
        assert isinstance(G.grammar, CFG)
        self.G = G
        self.cfg: CFG = G.grammar
        self._deleted: Set[Program] = set()

        # compute larger M
        all_costs = set(self.G.probabilities[S][P] for S in self.G.rules for P in self.G.rules[S] )
        self.basic_M = max(abs(x - y) for x in all_costs for y in all_costs)
        self.M = self.basic_M * (max(len(self.G.rules[S][P]) for S in self.G.rules for P in self.G.rules[S]) + 1) * len(self.G.rules)

        # Naming
        # nt -> one non terminal
        # derivation -> (S1, S2)

        self._queue_nt: Dict[Tuple[Type, U], List[Derivation]] = {}
        self._queue_derivation: Dict[Tuple[Tuple[Type, U]], CDQueue] = {}
        self._bank_nt: Dict[Tuple[Type, U], Dict[int, List[Program]]] = {}
        self._bank_derivation: Dict[Tuple[Tuple[Type, U]], Dict[int, List[List[List[Program]]]]] = {}
        self._cost_lists_nt: Dict[Tuple[Type, U], List[float]] = {}
        self._cost_lists_derivation: Dict[Tuple[Tuple[Type, U]], List[float]] = {}
        self._non_terminal_for: Dict[
            Tuple[Type, U], Dict[DerivableProgram, Tuple[Tuple[Type, U]]]
        ] = {}
        for S in self.G.grammar.rules:
            self._queue_nt[S] = []
            self._cost_lists_nt[S] = []
            self._bank_nt[S] = {}
            self._non_terminal_for[S] = {
                P: tuple([(Sp[0], (Sp[1], None)) for Sp in self.G.rules[S][P][0]])  # type: ignore
                for P in self.G.grammar.rules[S]
            }
            for P in self.G.grammar.rules[S]:
                args = self._non_terminal_for[S][P]
                if args and args not in self._queue_derivation:
                    self._queue_derivation[args] = CDQueue(self.M, k, precision)
                    self._bank_derivation[args] = {}
                    self._cost_lists_derivation[args] = []

    

    def _peek_next_derivation_cost_(self, S: Tuple[Type, U], P: DerivableProgram) -> float:
        args = self._non_terminal_for[S][P]
        if args:
            return self._queue_derivation[args].peek().cost
        return 0


    def _push_derivation_(self, S: Tuple[Type, U], P: DerivableProgram, element: CostTuple) -> None:
        args = self._non_terminal_for[S][P]
        return self._queue_derivation[args].push(element)

    def _pop_derivation_(self, S: Tuple[Type, U], P: DerivableProgram) -> CostTuple:
        args = self._non_terminal_for[S][P]
        return self._queue_derivation[args].pop()

    def _queue_size_(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        args = self._non_terminal_for[S][P]
        return self._queue_derivation[args].size()
    
    def _queue_is_empty_(self, S: Tuple[Type, U], P: DerivableProgram) -> bool:
        args = self._non_terminal_for[S][P]
        return self._queue_derivation[args].is_empty()

    def _init_derivation_(self, S: Tuple[Type, U], P: DerivableProgram) -> None:
        args = self._non_terminal_for[S][P]
        if len(self._cost_lists_derivation[args]) > 0:
            return
        cost = 0
        self._cost_lists_derivation[args].append(1e99)
        queue = self._queue_derivation[args]
        for Si in args:
            self._init_non_terminal_(Si)
            cost += self._cost_lists_nt[Si][0]
        index_cost = [0] * len(args)
        ct = CostTuple(cost, [index_cost])
        queue.push(ct)
        self._cost_lists_derivation[args][0] = queue.peek().cost
        


    def _init_non_terminal_(self, S: Tuple[Type, U]) -> None:
        if len(self._cost_lists_nt[S]) > 0:
            return
        self._cost_lists_nt[S].append(1e99)
        queue = self._queue_nt[S]
        for P in self.G.rules[S]:
            args = self._non_terminal_for[S][P]
            if args:
                self._init_derivation_(S, P)
                base_cost = self._peek_next_derivation_cost_(S, P)
            else:
                base_cost = 0
            heappush(queue, Derivation(base_cost + self.G.probabilities[S][P], 0, P))

        self._cost_lists_nt[S][0] = queue[0].cost

    def _reevaluate_derivation_(self, S: Tuple[Type, U], P: DerivableProgram) -> None:
        args = self._non_terminal_for[S][P]
        if args:
            queue = self._queue_derivation[args]
            elems = []
            while not queue.is_empty():
                elem = queue.pop()
                elems.append(CostTuple(sum(self._queue_nt[Si][0].cost for Si in args), elem.combinations))
            elems = sorted(elems)
            queue.clear()
            while not (len(elems) == 0):
                queue.push(elems.pop())
            self._cost_lists_derivation[args][0] = queue.peek().cost


    def _reevaluate_(self) -> None:
        if not self.cfg.is_recursive():
            return
        changed = True
        while changed:
            changed = False

            for S in list(self._queue_nt.keys()):
                for P in self.G.rules[S]:
                    self._reevaluate_derivation_(S, P)
                new_queue = [
                    Derivation(self.G.probabilities[S][el.P] + self._peek_next_derivation_cost_(S, el.P), el.combination, el.P)
                    for el in self._queue_nt[S]
                ]
                if new_queue != self._queue_nt[S]:
                    changed = True
                    heapify(new_queue)
                    self._queue_nt[S] = new_queue
                    self._cost_lists_nt[S][0] = self._queue_nt[S][0].cost

    def generator(self) -> Generator[Program, None, None]:
        self._init_non_terminal_(self.G.start)
        self._reevaluate_()
        # Update M
        values = set()
        for q in self._queue_nt.values():
            values |= set(el.cost for el in q)
        print("previous M:", self.M)
        self.M = max(self.basic_M, max(abs(x-y) for x in values for y in values))
        print("new M:", self.M)
        # Update queues
        for arg in self._queue_derivation:
            elems = []
            queue = self._queue_derivation[arg]
            # print("before:", queue)
            while not queue.is_empty():
                elems.append(queue.pop())
            self._queue_derivation[arg] = CDQueue(self.M, queue.k, queue.precision)
            elems = sorted(elems)
            while not (len(elems) == 0):
                self._queue_derivation[arg].push(elems.pop(0))
            assert len(self._queue_derivation[arg]) == 1

        n = 0
        failed = False
        while not failed:
            self._failed_by_empties = False
            failed = True
            for prog in self.query(self.G.start, n):
                failed = False
                yield prog
            failed = failed and not self._failed_by_empties
            n += 1

    def programs_in_banks(self) -> int:
        return sum(sum(len(x) for x in val.values()) for val in self._bank_nt.values())

    def programs_in_queues(self) -> int:
        return sum(len(val) for val in self._queue_nt.values()) + sum(cd.size() for cd in self._queue_derivation.values())

    def query_derivation(self, S: Tuple[Type, U], P: DerivableProgram, cost_index: int) -> List[List[List[Program]]]:
        args = self._non_terminal_for[S][P]
        if cost_index >= len(self._cost_lists_derivation[args]):
            return []
        # Memoization
        bank = self._bank_derivation[args]
        if cost_index in bank:
            return bank[cost_index]
        # Generation
        bank[cost_index] = []
        queue = self._queue_derivation[args]
        # print("before QUERY:", queue)

        if not queue.is_empty():
            ct = queue.pop()
            # print("\t\tAFTER POP:", queue)
            # print("\t\tpopping:", ct)
            for combination in ct.combinations:
                args_possibles = []
                for ci, Si in zip(combination, args):
                    allowed_empty, elems = self._query_list_(Si, ci)
                    if len(elems) == 0:
                        break
                    args_possibles.append(elems)
                if len(args_possibles) != len(combination):
                    continue
                bank[cost_index].append(args_possibles)

                # add successors
                for i in range(len(combination)):
                    cl = self._cost_lists_nt[args[i]]
                    if combination[i] + 1 >= len(cl):
                        if combination[i] + 1 > 1:
                            break
                        continue
                    index_cost = combination.copy()
                    index_cost[i] += 1
                    new_cost = ct.cost - cl[index_cost[i] - 1] + cl[index_cost[i]]
                    # print("\t\tpushing:", new_cost, ">", ct.cost, "cost tuple:", index_cost)
                    queue.push(CostTuple(new_cost, [index_cost]))
                    # print("\t\tAFTER PUSH:", queue)
                    # Avoid duplication with this condition
                    if index_cost[i] > 1:
                        break
            # print("after QUERY:", queue)
            # print(f"[BANK {args}][{cost_index}] = {bank[cost_index]}")
            if not queue.is_empty():
                queue.update()
                # print(queue)
                self._cost_lists_derivation[args].append(queue.peek().cost)
        return bank[cost_index]


    def query(
        self, S: Tuple[Type, U], cost_index: int
    ) -> Generator[Program, None, None]:
        # When we return this way, it actually mean that we have generated all programs that this non terminal could generate
        if cost_index >= len(self._cost_lists_nt[S]):
            return
        cost = self._cost_lists_nt[S][cost_index]
        bank = self._bank_nt[S]
        queue = self._queue_nt[S]
        while len(queue) > 0 and queue[0].cost == cost:
            element = heappop(queue)
            # print("[POP]:", element)
            if cost_index not in bank:
                bank[cost_index] = []
            args = self._non_terminal_for[S][element.P]
            if args:
                args_possibles = self.query_derivation(S, element.P, element.combination)
                # Finite nonterminal check
                if element.combination + 1 < len(self._cost_lists_derivation[args]):
                    next_cost = self.G.probabilities[S][element.P] + self._cost_lists_derivation[args][element.combination + 1]
                    heappush(queue, Derivation(next_cost, element.combination + 1, element.P))
                # Generate programs
                for possibles in args_possibles:
                    # print("S", S, "P", element.P, "index:", element.combination, "args:", possibles)
                    for new_args in product(*possibles):
                        new_program: Program = Function(element.P, list(new_args))
                        if new_program in self._deleted:
                            continue
                        elif not self._should_keep_subprogram(new_program):
                            self._deleted.add(new_program)
                            continue
                        bank[cost_index].append(new_program)
                        yield new_program
            else:
                new_program = element.P
                if new_program in self._deleted:
                    continue
                elif not self._should_keep_subprogram(new_program):
                    self._deleted.add(new_program)
                    continue
                bank[cost_index].append(new_program)
                yield new_program

        if len(queue) > 0:
            next_cost = queue[0].cost
            self._cost_lists_nt[S].append(next_cost)

    def _query_list_(
        self, S: Tuple[Type, U], cost_index: int
    ) -> Tuple[bool, List[Program]]:
        """
        returns is_allowed_empty, programs
        """
        # It's an empty cost index but a valid one
        # if cost_index in self._empties[S]:
            # return True, []
        if cost_index >= len(self._cost_lists_nt[S]):
            return False, []
        bank = self._bank_nt[S]
        if cost_index in bank:
            return False, bank[cost_index]
        for x in self.query(S, cost_index):
            pass
        # if cost_index in self._empties[S]:
            # return True, []
        return False, bank[cost_index]

    def merge_program(self, representative: Program, other: Program) -> None:
        self._deleted.add(other)
        for S in self.G.rules:
            if S[0] != other.type:
                continue
            local_bank = self._bank_nt[S]
            for programs in local_bank.values():
                if other in programs:
                    programs.remove(other)

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "cd-search"

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "CDSearch[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        enum._deleted = self._deleted.copy()
        return enum


def enumerate_prob_grammar(G: ProbDetGrammar[U, V, W], k: int = 10, precision: float = 1e-5) -> CDSearch[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: -np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return CDSearch(Gp, k=k, precision=precision)
