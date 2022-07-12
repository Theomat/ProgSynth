from typing import (
    Dict,
    Generator,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import vose

from synth.syntax.grammars.det_grammar import DerivableProgram, DetGrammar
from synth.syntax.program import Function, Program
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class ProbDetGrammar(DetGrammar[U, V, Tuple[W, float]]):
    def __init__(
        self,
        grammar: DetGrammar[U, V, W],
        probabilities: Dict[Tuple[Type, U], Dict[DerivableProgram, float]],
    ):
        super().__init__(grammar.start, grammar.rules, clean=False)
        self.grammar = grammar
        self.probabilities = probabilities
        self.ready_for_sampling = False

    def __hash__(self) -> int:
        return hash((self.start, self.grammar, self.probabilities))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, ProbDetGrammar)
            and self.grammar == o.grammar
            and self.probabilities == o.probabilities
        )

    def name(self) -> str:
        return "P" + self.grammar.name()

    def __str__(self) -> str:
        s = f"Print a P{self.grammar.__class__.__qualname__}\n"
        s += "start: {}\n".format(self.grammar.start)
        for S in reversed(self.grammar.rules):
            s += "#\n {}\n".format(S)
            for P in self.grammar.rules[S]:
                out = self.grammar.rules[S][P]
                s += "   {} ~> {}\n".format(
                    self.probabilities[S][P], self.grammar.__rule_to_str__(P, out)
                )
        return s

    def derive(
        self, information: Tuple[W, float], S: Tuple[Type, U], P: DerivableProgram
    ) -> Tuple[Tuple[W, float], Tuple[Type, U]]:
        base_information, new_ctx = self.grammar.derive(information[0], S, P)
        prob = self.probabilities[S][P]
        old_prob = information[1]
        return ((base_information, prob * old_prob), new_ctx)

    def start_information(self) -> Tuple[W, float]:
        return (self.grammar.start_information(), 1)

    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        return self.grammar.arguments_length_for(S, P)

    def _remove_non_productive_(self) -> None:
        pass

    def _remove_non_reachable_(self) -> None:
        pass

    def probability(
        self,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> float:
        return self.reduce_derivations(
            lambda current, S, P, _: current * self.probabilities[S][P],
            1.0,
            program,
            start,
        )

    def init_sampling(self, seed: Optional[int] = None) -> None:
        """
        seed = 0 <=> No Seeding
        """
        self.ready_for_sampling = True
        self.vose_samplers = {}
        self.sampling_map = {}

        for i, S in enumerate(self.rules):
            P_list = list(self.probabilities[S].keys())
            self.vose_samplers[S] = vose.Sampler(
                np.array(
                    [self.probabilities[S][P] for P in P_list],
                    dtype=float,
                ),
                seed=seed + i if seed else None,
            )
            self.sampling_map[S] = P_list

    def normalise(self) -> None:
        for S in self.probabilities:
            s = sum(self.probabilities[S][P] for P in self.probabilities[S])
            for P in list(self.probabilities[S].keys()):
                w = self.probabilities[S][P]
                self.probabilities[S][P] = w / s

    def sampling(self) -> Generator[Program, None, None]:
        """
        A generator that samples programs according to the PCFG G
        """
        assert self.ready_for_sampling
        print(self)
        while True:
            print("SAMPLING A PROGRAM")
            yield self.sample_program(self.start)
            print("DONE")

    def sample_program(
        self, S: Optional[Tuple[Type, U]] = None, information: Optional[W] = None
    ) -> Program:
        assert self.ready_for_sampling
        S = S or self.start
        i: int = self.vose_samplers[S].sample()
        P = self.sampling_map[S][i]
        nargs = self.arguments_length_for(S, P)
        if nargs == 0:
            return P
        arguments = []
        information = information or self.grammar.start_information()
        information, current = self.grammar.derive(information, S, P)
        for _ in range(nargs):
            arg = self.sample_program(current, information)
            arguments.append(arg)
            information, lst = self.grammar.derive_all(information, current, arg)
            current = lst[-1]
        return Function(P, arguments)

    @classmethod
    def uniform(cls, grammar: DetGrammar[U, V, W]) -> "ProbDetGrammar[U, V, W]":
        return ProbDetGrammar(
            grammar,
            {
                S: {_: 1 / len(grammar.rules[S]) for _ in grammar.rules[S]}
                for S in grammar.rules
            },
        )
