from math import prod
from typing import Dict, Generator, List, Optional, Set, Tuple, TypeVar, Generic

import numpy as np

import vose

from synth.syntax.concrete.concrete_cfg import NonTerminal
from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow

T = TypeVar("T")


class UPCFG(Generic[T]):
    """
    Represents an unambiguous, possibly non deterministic PCFG.
    """

    def __init__(
        self,
        start: Tuple[NonTerminal, T],
        rules: Dict[
            Tuple[NonTerminal, T],
            Dict[Tuple[Program, T], Tuple[List[Tuple[NonTerminal, T]], float]],
        ],
        max_program_depth: int,
        clean: bool = False,
    ):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth
        self.hash_table_programs: Dict[int, Program] = {}

        if clean:
            self.clean()

        # Compute the type request
        type_req = self.start[0].type
        variables: List[Variable] = []
        for S in self.rules:
            for P, _ in self.rules[S]:
                if isinstance(P, Variable):
                    if P not in variables:
                        variables.append(P)
        n = len(variables)
        for i in range(n):
            j = n - i - 1
            for v in variables:
                if v.variable == j:
                    type_req = Arrow(v.type, type_req)
        self.type_request = type_req
        self.ready_for_sampling: bool = False

    def clean(self) -> None:
        """
        Remove non productive.
        Remove non reachable
        Normalise probabilities
        Sort rules
        """
        self.__remove_non_reachable__()
        self.__normalise__()
        self.__sort__()

    def __hash__(self) -> int:
        return hash((self.start, str(self.rules), self.max_program_depth))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, UPCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __str__(self) -> str:
        s = "Print an NPCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P[0], P[0].type, args_P, w)
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def init_sampling(self, seed: Optional[int] = None) -> None:
        self.ready_for_sampling = True
        self.vose_samplers = {}
        self.list_derivations = {}

        for i, S in enumerate(self.rules):
            self.list_derivations[S] = sorted(
                self.rules[S], key=lambda P: self.rules[S][P][1]
            )
            self.vose_samplers[S] = vose.Sampler(
                np.array(
                    [self.rules[S][P][1] for P in self.list_derivations[S]], dtype=float
                ),
                seed=seed + i if seed else None,
            )

    def __sort__(self) -> None:
        for S in self.rules:
            sorted_derivation_list = sorted(
                self.rules[S], key=lambda P: -self.rules[S][P][1]
            )
            new_rules = {}
            for P in sorted_derivation_list:
                new_rules[P] = self.rules[S][P]
            self.rules[S] = new_rules

    def __normalise__(self) -> None:
        for S in self.rules:
            s = sum([self.rules[S][P][1] for P in self.rules[S]])
            for P in list(self.rules[S].keys()):
                args_P, w = self.rules[S][P]
                self.rules[S][P] = (args_P, w / s)

    def return_unique(self, P: Program) -> Program:
        """
        ensures that if a program appears in several rules,
        it is represented by the same object
        """
        h = hash(P)
        if h in self.hash_table_programs:
            return self.hash_table_programs[h]
        else:
            self.hash_table_programs[h] = P
            return P

    def __remove_non_reachable__(self) -> None:
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable: Set[Tuple[NonTerminal, T]] = set()
        reachable.add(self.start)

        reach: Set[Tuple[NonTerminal, T]] = set()
        new_reach: Set[Tuple[NonTerminal, T]] = set()
        reach.add(self.start)

        for _ in range(self.max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P, _ = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def sampling(self) -> Generator[Program, None, None]:
        """
        A generator that samples programs according to the PCFG G
        """
        assert self.ready_for_sampling
        while True:
            yield self.sample_program(self.start[0], self.start[1])

    def sample_program(
        self, S: Optional[NonTerminal] = None, remaining_score: Optional[T] = None
    ) -> Program:
        assert self.ready_for_sampling
        S = S or self.start[0]
        score = remaining_score or self.start[1]
        C = (S, score)
        i: int = self.vose_samplers[C].sample()
        P = self.list_derivations[C][i]
        args_P, _ = self.rules[C][P]
        if len(args_P) == 0:
            return P[0]
        arguments = []
        for arg in args_P:
            arguments.append(self.sample_program(arg[0], arg[1]))
        return Function(P[0], arguments)

    def __contains__(
        self, P: Program, S: Optional[Tuple[NonTerminal, T]] = None
    ) -> bool:
        S = S or self.start
        if S not in self.rules:
            return False
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            possibles = [(P_prime, i) for P_prime, i in self.rules[S] if P_prime == F]
            for possible in possibles:
                contained = all(
                    self.__contains__(arg, self.rules[S][possible][0][i])
                    for i, arg in enumerate(args_P)
                )
                if contained:
                    return True
            return False

        elif isinstance(P, (Variable, Primitive)):
            return any(True for P_prime, _ in self.rules[S] if P_prime == P)

        return False

    def probability(
        self, P: Program, S: Optional[Tuple[NonTerminal, T]] = None
    ) -> float:
        """
        Compute the probability of a program P generated from the non-terminal S
        """
        S = S or self.start
        if S not in self.rules:
            return 0
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            possibles = [(P_prime, i) for P_prime, i in self.rules[S] if P_prime == F]
            p_F = sum(self.rules[S][(P_prime, i)][1] for P_prime, i in possibles)
            prob = 0.0
            for possible in possibles:
                prob += prod(
                    self.probability(arg, self.rules[S][possible][0][i])
                    for i, arg in enumerate(args_P)
                )
            return prob * p_F

        elif isinstance(P, (Variable, Primitive)):
            return sum(
                self.rules[S][(P_prime, i)][1]
                for P_prime, i in self.rules[S]
                if P_prime == P
            )
        return 0
