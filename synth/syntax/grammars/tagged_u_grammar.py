from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import vose
from synth.syntax.grammars.cfg import CFG, CFGNonTerminal, CFGState, NoneType

from synth.syntax.grammars.det_grammar import DerivableProgram, DetGrammar
from synth.syntax.grammars.u_grammar import UGrammar
from synth.syntax.program import Function, Program
from synth.syntax.type_system import Type

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class TaggedUGrammar(UGrammar[U, V, W], Generic[T, U, V, W]):
    def __init__(
        self,
        grammar: UGrammar[U, V, W],
        tags: Dict[Tuple[Type, U], Dict[DerivableProgram, Dict[V, T]]],
    ):
        super().__init__(grammar.start, grammar.rules, clean=False)
        self.grammar = grammar
        self.tags = tags

    def __hash__(self) -> int:
        return hash((self.start, self.grammar, str(self.tags)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TaggedUGrammar)
            and self.grammar == o.grammar
            and self.tags == o.tags
        )

    def name(self) -> str:
        return "tagged" + self.grammar.name()

    def __str__(self) -> str:
        s = f"Print a {self.name()}\n"
        s += "start: {}\n".format(self.grammar.start)
        for S in reversed(self.grammar.rules):
            s += "#\n {}\n".format(S)
            for P in self.grammar.rules[S]:
                out = self.grammar.rules[S][P]
                for possible in out:
                    s += "   {} ~> {}\n".format(
                        self.tags[S][P][possible],
                        self.grammar.__rule_to_str__(P, possible),
                    )
        return s

    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        return self.grammar.arguments_length_for(S, P)

    def derive(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram
    ) -> List[Tuple[W, Tuple[Type, U], V]]:
        return self.grammar.derive(information, S, P)

    def start_information(self) -> W:
        return self.grammar.start_information()

    def __add__(
        self, other: "TaggedUGrammar[T, U, V, W]"
    ) -> "TaggedUGrammar[T, U, V, W]":
        new_probs: Dict[Tuple[Type, U], Dict[DerivableProgram, Dict[V, T]]] = {}
        for S in set(self.tags.keys()).union(other.tags.keys()):
            new_probs[S] = {}
            for P in set(self.tags.get(S, {})).union(other.tags.get(S, {})):
                for key in self.tags[S][P]:
                    new_probs[S][P][key] = self.tags[S][P][key] + other.tags[S][P][key]  # type: ignore
        return TaggedUGrammar(self.grammar, new_probs)


class ProbUGrammar(TaggedUGrammar[float, U, V, W]):
    def __init__(
        self,
        grammar: UGrammar[U, V, W],
        probabilities: Dict[Tuple[Type, U], Dict[DerivableProgram, Dict[V, float]]],
    ):
        super().__init__(grammar, probabilities)
        self.ready_for_sampling = False

    @property
    def probabilities(
        self,
    ) -> Dict[Tuple[Type, U], Dict[DerivableProgram, Dict[V, float]]]:
        return self.tags

    def name(self) -> str:
        return "P" + self.grammar.name()

    def __mul__(self, other: float) -> "ProbUGrammar[U, V, W]":
        return ProbUGrammar(
            self.grammar,
            {
                S: {P: {v: p * other for v, p in lst.items()} for P, lst in v.items()}
                for S, v in self.tags.items()
            },
        )

    def __rmul__(self, other: float) -> "ProbUGrammar[U, V, W]":
        return self.__mul__(other)

    def probability(
        self,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> float:
        try:
            return self.reduce_derivations(
                lambda current, S, P, V: current * self.tags[S][P][V],
                1.0,
                program,
                start,
            )[0]
        except:
            return 0

    def init_sampling(self, seed: Optional[int] = None) -> None:
        """
        seed = 0 <=> No Seeding
        """
        self.ready_for_sampling = True
        self.vose_samplers: Dict[Tuple[Type, U], Any] = {}
        self.sampling_map: Dict[Tuple[Type, U], List[DerivableProgram]] = {}
        self._vose_samplers_2: Dict[Tuple[Type, U], Dict[DerivableProgram, Any]] = {}

        for i, S in enumerate(self.tags):
            P_list = list(self.tags[S].keys())
            self.vose_samplers[S] = vose.Sampler(
                np.array(
                    [sum(p for p in self.tags[S][P].values()) for P in P_list],
                    dtype=float,
                ),
                seed=seed + i if seed else None,
            )
            self._vose_samplers_2[S] = {}
            for P in P_list:
                self._vose_samplers_2[S][P] = vose.Sampler(
                    np.array(
                        [p for p in self.tags[S][P].values()],
                        dtype=float,
                    ),
                    seed=seed + i if seed else None,
                )
            self.sampling_map[S] = P_list

    def normalise(self) -> None:
        for S in self.tags:
            s = sum(
                sum(self.tags[S][P][V] for V in self.tags[S][P]) for P in self.tags[S]
            )
            for P in list(self.tags[S].keys()):
                w = self.tags[S][P]
                self.tags[S][P] = {v: p / s for v, p in w.items()}

    def sampling(self) -> Generator[Program, None, None]:
        """
        A generator that samples programs according to the PCFG G
        """
        assert self.ready_for_sampling
        while True:
            yield self.sample_program(self.start)

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
        i = self._vose_samplers_2[S][P].sample()
        information, current, _ = self.grammar.derive(information, S, P)[i]
        for _ in range(nargs):
            arg = self.sample_program(current, information)
            arguments.append(arg)
            information, lst = self.grammar.derive_all(information, current, arg)[0]
            current = lst[-1][0]
        return Function(P, arguments)

    @classmethod
    def uniform(cls, grammar: UGrammar[U, V, W]) -> "ProbUGrammar[U, V, W]":
        probs: Dict[Tuple[Type, U], Dict[DerivableProgram, Dict[V, float]]] = {}
        for S in grammar.rules:
            probs[S] = {}
            n = sum(len(grammar.rules[S][P]) for P in grammar.rules[S])
            for P in grammar.rules[S]:
                probs[S][P] = {v: 1 / n for v in grammar.rules[S][P]}
        return ProbUGrammar(
            grammar,
            probs,
        )
