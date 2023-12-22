from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)

import numpy as np
import vose

if TYPE_CHECKING:
    from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.grammar import NGram
from synth.syntax.grammars.det_grammar import DerivableProgram, DetGrammar
from synth.syntax.program import Constant, Function, Program
from synth.syntax.type_system import Type

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

NoneType = Literal[None]
CFGState = Tuple[NGram, int]
CFGNonTerminal = Tuple[Type, Tuple[CFGState, NoneType]]


class TaggedDetGrammar(DetGrammar[U, V, W], Generic[T, U, V, W]):
    def __init__(
        self,
        grammar: DetGrammar[U, V, W],
        tags: Dict[Tuple[Type, U], Dict[DerivableProgram, T]],
    ):
        super().__init__(grammar.start, grammar.rules, clean=False)
        self.grammar = grammar
        self.tags = tags

    def programs(self) -> int:
        return self.grammar.programs()

    def __hash__(self) -> int:
        return hash((self.start, self.grammar, str(self.tags)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TaggedDetGrammar)
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
                s += "   {} ~> {}\n".format(
                    self.tags[S][P], self.grammar.__rule_to_str__(P, out)
                )
        return s

    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        return self.grammar.arguments_length_for(S, P)

    def derive(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram
    ) -> Tuple[W, Tuple[Type, U]]:
        return self.grammar.derive(information, S, P)

    def start_information(self) -> W:
        return self.grammar.start_information()

    def __add__(
        self, other: "TaggedDetGrammar[T, U, V, W]"
    ) -> "TaggedDetGrammar[T, U, V, W]":
        new_probs: Dict[Tuple[Type, U], Dict[DerivableProgram, T]] = {}
        for S in set(self.tags.keys()).union(other.tags.keys()):
            new_probs[S] = {}
            for P in set(self.tags.get(S, {})).union(other.tags.get(S, {})):
                if S in self.tags and P in self.tags[S]:
                    safe = {P: self.tags[S][P]}
                else:
                    safe = {P: other.tags[S][P]}
                new_probs[S][P] = self.tags.get(S, safe)[P] + other.tags.get(S, safe)[P]  # type: ignore
        return self.__class__(self.grammar, new_probs)

    def instantiate_constants(
        self, constants: Dict[Type, List[Any]]
    ) -> "TaggedDetGrammar[T, U, V, W]":
        tags: Dict[Tuple[Type, U], Dict[DerivableProgram, T]] = {}

        for S in self.tags:
            tags[S] = {}
            for P in self.tags[S]:
                if isinstance(P, Constant) and P.type in constants:
                    for val in constants[P.type]:
                        tags[S][Constant(P.type, val, True)] = self.tags[S][P]
                else:
                    tags[S][P] = self.tags[S][P]
        return self.__class__(self.grammar.instantiate_constants(constants), tags)


class ProbDetGrammar(TaggedDetGrammar[float, U, V, W]):
    def __init__(
        self,
        grammar: DetGrammar[U, V, W],
        probabilities: Dict[Tuple[Type, U], Dict[DerivableProgram, float]],
    ):
        super().__init__(grammar, probabilities)
        self.ready_for_sampling = False

    def __hash__(self) -> int:
        return super().__hash__()

    @property
    def probabilities(self) -> Dict[Tuple[Type, U], Dict[DerivableProgram, float]]:
        return self.tags

    def name(self) -> str:
        return "P" + self.grammar.name()

    def __mul__(self, other: float) -> "ProbDetGrammar[U, V, W]":
        return ProbDetGrammar(
            self.grammar,
            {S: {P: other * p for P, p in v.items()} for S, v in self.tags.items()},
        )

    def __rmul__(self, other: float) -> "ProbDetGrammar[U, V, W]":
        return self.__mul__(other)

    def probability(
        self,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> float:
        try:
            return self.reduce_derivations(
                lambda current, S, P, _: current * self.tags[S][P],
                1.0,
                program,
                start,
            )
        except:
            return 0

    def init_sampling(self, seed: Optional[int] = None) -> None:
        """
        seed = 0 <=> No Seeding
        """
        self.ready_for_sampling = True
        self.vose_samplers = {}
        self.sampling_map = {}

        for i, S in enumerate(self.tags):
            P_list = list(self.tags[S].keys())
            self.vose_samplers[S] = vose.Sampler(
                np.array(
                    [self.tags[S][P] for P in P_list],
                    dtype=float,
                ),
                seed=seed + i if seed else None,
            )
            self.sampling_map[S] = P_list

    def normalise(self) -> None:
        for S in self.tags:
            s = sum(self.tags[S][P] for P in self.tags[S])
            for P in list(self.tags[S].keys()):
                w = self.tags[S][P]
                self.tags[S][P] = w / s

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
        information, current = self.grammar.derive(information, S, P)
        for _ in range(nargs):
            arg = self.sample_program(current, information)
            arguments.append(arg)
            information, lst = self.grammar.derive_all(information, current, arg)
            current = lst[-1]
        return Function(P, arguments)

    def instantiate_constants(
        self, constants: Dict[Type, List[Any]]
    ) -> "ProbDetGrammar[U, V, W]":
        tags: Dict[Tuple[Type, U], Dict[DerivableProgram, float]] = {}

        for S in self.tags:
            tags[S] = {}
            for P in self.tags[S]:
                if isinstance(P, Constant) and P.type in constants:
                    for val in constants[P.type]:
                        tags[S][Constant(P.type, val, True)] = self.tags[S][P] / len(
                            constants[P.type]
                        )
                else:
                    tags[S][P] = self.tags[S][P]
        return self.__class__(self.grammar.instantiate_constants(constants), tags)

    @classmethod
    def uniform(cls, grammar: DetGrammar[U, V, W]) -> "ProbDetGrammar[U, V, W]":
        return ProbDetGrammar(
            grammar,
            {
                S: {_: 1 / len(grammar.rules[S]) for _ in grammar.rules[S]}
                for S in grammar.rules
            },
        )

    @classmethod
    def random(
        cls,
        grammar: DetGrammar[U, V, W],
        seed: Optional[int] = None,
        gen: Callable[[np.random.Generator], float] = lambda prng: prng.uniform(),
    ) -> "ProbDetGrammar[U, V, W]":
        prng = np.random.default_rng(seed)
        pg = ProbDetGrammar(
            grammar,
            {S: {_: gen(prng) for _ in grammar.rules[S]} for S in grammar.rules},
        )
        pg.normalise()
        return pg

    @classmethod
    def pcfg_from_samples(
        cls, cfg: "CFG", samples: Iterable[Program]
    ) -> "ProbDetGrammar[Tuple[CFGState, NoneType], Tuple[List[Tuple[Type, CFGState]], NoneType], List[Tuple[Type, CFGState]]]":
        rules_cnt: Dict[CFGNonTerminal, Dict[DerivableProgram, int]] = {}
        for S in cfg.rules:
            rules_cnt[S] = {}
            for P in cfg.rules[S]:
                rules_cnt[S][P] = 0

        def add_count(S: CFGNonTerminal, P: Program) -> bool:
            if isinstance(P, Function):
                F = P.function
                args_P = P.arguments
                success = add_count(S, F)

                args = cfg.rules[S][F][0]  # type: ignore
                for i, arg in enumerate(args_P):
                    add_count((args[i][0], (args[i][1], None)) if success else S, arg)
            else:
                if P not in rules_cnt[S]:
                    # This case occurs when a forbidden pattern has been removed from the CFG
                    # What to do? Ignore for now, but this bias a bit the probabilities
                    # TODO: perhaps rethink that? or provide a program simplifier
                    return False
                else:
                    rules_cnt[S][P] += 1  # type: ignore
            return True

        for sample in samples:
            add_count(cfg.start, sample)

        # Compute probabilities
        probabilities: Dict[CFGNonTerminal, Dict[DerivableProgram, float]] = {}
        for S in cfg.rules:
            total = sum(rules_cnt[S][P] for P in cfg.rules[S])
            if total > 0:
                probabilities[S] = {}
                for P in rules_cnt[S]:
                    probabilities[S][P] = rules_cnt[S][P] / total

        return ProbDetGrammar(cfg, probabilities)
