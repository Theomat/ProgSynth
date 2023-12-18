from collections import defaultdict, deque
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Union,
    overload,
)

from synth.syntax.dsl import DSL
from synth.syntax.automata.dfa import DFA
from synth.syntax.grammars.grammar import DerivableProgram, NGram
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.program import Constant, Primitive, Variable
from synth.syntax.type_system import Arrow, Type, UnknownType
from synth.syntax.grammars.det_grammar import DetGrammar

T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")
V = TypeVar("V")


class TTCFG(
    DetGrammar[Tuple[S, T], Tuple[List[Tuple[Type, S]], T], List[Tuple[Type, S]]],
    Generic[S, T],
):
    """
    Represents a deterministic Tree Traversing CFG (TTCFG).
    """

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TTCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __hash__(self) -> int:
        return super().__hash__()

    def derive(
        self,
        information: List[Tuple[Type, S]],
        start: Tuple[Type, Tuple[S, T]],
        program: DerivableProgram,
    ) -> Tuple[List[Tuple[Type, S]], Tuple[Type, Tuple[S, T]]]:
        args, state = self.rules[start][program]
        if args:
            information = args + information
            nrule = (information[0][0], (information[0][1], state))
            return information[1:], nrule
        elif information:
            nrule = (information[0][0], (information[0][1], state))
            return information[1:], nrule
        # This will cause an error if this is not the last call to derive
        return information, (UnknownType(), (start[1][0], state))

    @overload
    def __mul__(self, other: "TTCFG[U, V]") -> "TTCFG[Tuple[S, U], Tuple[T, V]]":
        pass

    @overload
    def __mul__(self, other: DFA[U, DerivableProgram]) -> "TTCFG[S, Tuple[T, U]]":
        pass

    @overload
    def __mul__(
        self, other: DFA[U, Tuple[Tuple[Type, Tuple[S, T]], DerivableProgram]]
    ) -> "TTCFG[S, Tuple[T, U]]":
        pass

    def __mul__(
        self,
        other: Union[
            "TTCFG[U, V]",
            DFA[U, DerivableProgram],
            DFA[U, Tuple[Tuple[Type, Tuple[S, T]], DerivableProgram]],
        ],
    ) -> Union["TTCFG[S, Tuple[T, U]]", "TTCFG[Tuple[S, U], Tuple[T, V]]"]:
        if isinstance(other, TTCFG):
            return self.__mul_ttcfg__(other)
        elif isinstance(other, DFA):
            if isinstance(list(other.rules[other.start].keys())[0], tuple):
                return self.__mul_dfa__(other)  # type: ignore
            else:
                return self.__mul_dfa_simple__(other)  # type: ignore
        assert False, f"Cannot multiply TTCFG with {other}"

    def __mul_ttcfg__(self, other: "TTCFG[U, V]") -> "TTCFG[Tuple[S, U], Tuple[T, V]]":
        assert (
            self.type_request == other.type_request
        ), "Both TTCFGs do not have the same type request!"
        rules: Dict[
            Tuple[Type, Tuple[Tuple[S, U], Tuple[T, V]]],
            Dict[
                Union[Primitive, Variable, Constant],
                Tuple[List[Tuple[Type, Tuple[S, U]]], Tuple[T, V]],
            ],
        ] = {}
        start: Tuple[Type, Tuple[Tuple[S, U], Tuple[T, V]]] = (
            self.start[0],
            (
                (self.start[1][0], other.start[1][0]),
                (self.start[1][1], other.start[1][1]),
            ),
        )
        for nT1 in self.rules:
            for nT2 in other.rules:
                # check type equality
                if nT1[0] != nT2[0]:
                    continue
                rule = (nT1[0], ((nT1[1][0], nT2[1][0]), (nT1[1][1], nT2[1][1])))
                rules[rule] = {}
                for P1 in self.rules[nT1]:
                    for P2 in other.rules[nT2]:
                        if P1 != P2:
                            continue
                        new_deriv = [
                            (el1[0], (el1[1], el2[1]))
                            for el1, el2 in zip(
                                self.rules[nT1][P1][0], other.rules[nT2][P1][0]
                            )
                        ]
                        rules[rule][P1] = (
                            new_deriv,
                            (self.rules[nT1][P1][1], other.rules[nT2][P1][1]),
                        )

        return TTCFG(start, rules, clean=True)

    def __mul_dfa_simple__(
        self, other: DFA[U, DerivableProgram]
    ) -> "TTCFG[S, Tuple[T, U]]":
        rules: Dict[
            Tuple[Type, Tuple[S, Tuple[T, U]]],
            Dict[
                Union[Primitive, Variable, Constant],
                Tuple[List[Tuple[Type, S]], Tuple[T, U]],
            ],
        ] = {}
        start: Tuple[Type, Tuple[S, Tuple[T, U]]] = (
            self.start[0],
            (
                self.start[1][0],
                (self.start[1][1], other.start),
            ),
        )
        for nT1 in self.rules:
            for nT2 in other.rules:
                rule = (nT1[0], (nT1[1][0], (nT1[1][1], nT2)))
                rules[rule] = {}
                for P1 in self.rules[nT1]:
                    for P2 in other.rules[nT2]:
                        if P1 != P2:
                            continue
                        new_deriv = self.rules[nT1][P1][0][:]
                        rules[rule][P1] = (
                            new_deriv,
                            (self.rules[nT1][P1][1], other.rules[nT2][P2]),
                        )
        return TTCFG(start, rules, clean=True)

    def __mul_dfa__(
        self, other: DFA[U, Tuple[Tuple[Type, Tuple[S, T]], DerivableProgram]]
    ) -> "TTCFG[S, Tuple[T, U]]":
        rules: Dict[
            Tuple[Type, Tuple[S, Tuple[T, U]]],
            Dict[
                Union[Primitive, Variable, Constant],
                Tuple[List[Tuple[Type, S]], Tuple[T, U]],
            ],
        ] = {}
        start: Tuple[Type, Tuple[S, Tuple[T, U]]] = (
            self.start[0],
            (
                self.start[1][0],
                (self.start[1][1], other.start),
            ),
        )
        for nT1 in self.rules:
            for nT2 in other.rules:
                rule = (nT1[0], (nT1[1][0], (nT1[1][1], nT2)))
                rules[rule] = {}
                for S2, P2 in other.rules[nT2]:
                    if S2 != nT1:
                        continue
                    for P1 in self.rules[nT1]:
                        if P1 != P2:
                            continue
                        new_deriv = self.rules[nT1][P1][0][:]
                        rules[rule][P1] = (
                            new_deriv,
                            (self.rules[nT1][P1][1], other.rules[nT2][(S2, P2)]),
                        )
        return TTCFG(start, rules, clean=True)

    def clean(self) -> None:
        # 1) Only keep reachable states
        new_rules: Dict[Tuple[Type, Tuple[S, T]], Set[DerivableProgram]] = {}
        list_to_be_treated: Deque[
            Tuple[Tuple[Type, Tuple[S, T]], List[Tuple[Type, S]]]
        ] = deque()

        list_to_be_treated.append((self.start, self.start_information()))
        while list_to_be_treated:
            rule, info = list_to_be_treated.pop()
            if rule not in new_rules:
                new_rules[rule] = set()
            # Create rule if non existent
            for P in self.rules[rule]:
                new_rules[rule].add(P)
                new_info, new_S = self.derive(info, rule, P)
                if new_S in self.rules:
                    list_to_be_treated.append((new_S, new_info))

        # 2) Remove empty non terminals
        def clean() -> bool:
            list_to_be_treated: Deque[
                Tuple[Tuple[Type, Tuple[S, T]], List[Tuple[Type, S]]]
            ] = deque()
            return_value = False
            list_to_be_treated.append((self.start, self.start_information()))
            while list_to_be_treated:
                rule, info = list_to_be_treated.pop()
                if rule not in new_rules:
                    continue
                if len(new_rules[rule]) == 0:
                    del new_rules[rule]
                    return_value = True
                    continue
                # Create rule if non existent
                for P in list(new_rules[rule]):
                    new_info, new_S = self.derive(info, rule, P)
                    if (
                        new_S not in new_rules
                        and new_S in self.rules
                        and len(new_info) >= len(info)
                    ):
                        new_rules[rule].remove(P)
                        if len(new_rules[rule]) == 0:
                            del new_rules[rule]
                            return_value = True
                    elif new_S in self.rules:
                        list_to_be_treated.append((new_S, new_info))
            return return_value

        while clean():
            pass

        self.rules = {S: {P: self.rules[S][P] for P in new_rules[S]} for S in new_rules}

    def start_information(self) -> List[Tuple[Type, S]]:
        return []

    def arguments_length_for(
        self, S: Tuple[Type, Tuple[S, T]], P: DerivableProgram
    ) -> int:
        return len(self.rules[S][P][0])

    def __rule_to_str__(
        self, P: DerivableProgram, out: Tuple[List[Tuple[Type, S]], T]
    ) -> str:
        args, state = out
        return "{}: {}\t\t{}".format(P, state, args)

    def name(self) -> str:
        return "TTCFG"

    @lru_cache()
    def programs(self) -> int:
        """
        Return the total number of programs contained in this grammar.
        """
        _counts: Dict[Tuple[Type, Tuple[S, T]], Dict[T, int]] = {}

        def __compute__(state: Tuple[Type, Tuple[S, T]]) -> Dict[T, int]:
            if state in _counts:
                return _counts[state]
            if state not in self.rules:
                return {state[1][1]: 1}
            output: Dict[T, int] = defaultdict(int)
            for P in self.rules[state]:
                info, new_state = self.derive(self.start_information(), state, P)
                local = __compute__(new_state)
                while info:
                    base = info.pop()
                    next_local: Dict[T, int] = defaultdict(int)
                    for v, cnt in local.items():
                        next_new_state = (base[0], (base[1], v))
                        for nV, nC in __compute__(next_new_state).items():
                            next_local[nV] += nC * cnt
                    local = next_local
                for v, c in local.items():
                    output[v] += c
            _counts[state] = output
            return output

        return sum(c for _, c in __compute__(self.start).items())

    def programs_stochastic(
        self, cfg: "TTCFG", samples: int = 10000, seed: Optional[int] = None
    ) -> float:
        """
        Provides an estimate of the number of programs in this grammar based on cfg.
        cfg must contain this grammar.
        Returns: the fraction of programs of cfg that this grammar contains
        """
        pcfg = ProbDetGrammar.uniform(cfg)
        pcfg.init_sampling(seed)
        inside = 0
        for _ in range(samples):
            if pcfg.sample_program() in self:
                inside += 1
        return inside / samples

    def possible_outcomes_after(
        self,
        S: Tuple[Type, Tuple[S, T]],
        P: Optional[DerivableProgram] = None,
        info: Optional[List[Tuple[Type, S]]] = None,
    ) -> Set[T]:
        """
        Return the set of all possibles T that can be generated starting from S -> P or just from S.
        """
        info = info or self.start_information()
        if S not in self.rules:
            return set()
        if P is None:
            # print(f"  ask {S} ({info})")
            out_plus = set()
            for P in self.rules[S]:
                out_plus |= self.possible_outcomes_after(S, P, info)
            # print(f"  end {S} ({info})")

            return out_plus
        new_info, new_S = self.derive(info, S, P)
        if new_S not in self.rules:
            assert (
                len(new_info) == 0
            ), f"info:{new_info} from {S}->{P} ({info}) obtained: {new_S}"
            return set([new_S[1][1]])
        # print(f"    ask {S} -> {P} ({info}->{new_info})")
        # Derive all possible children
        out = set()
        for PP in self.rules[new_S]:
            candidates = self.possible_outcomes_after(new_S, PP, new_info)
            out |= candidates
        return out

    def instantiate_constants(self, constants: Dict[Type, List[Any]]) -> "TTCFG[S, T]":
        rules: Dict[
            Tuple[Type, Tuple[S, T]],
            Dict[DerivableProgram, Tuple[List[Tuple[Type, S]], T]],
        ] = {}
        for NT in self.rules:
            rules[NT] = {}
            for P in self.rules[NT]:
                if isinstance(P, Constant) and P.type in constants:
                    for val in constants[P.type]:
                        rules[NT][Constant(P.type, val, True)] = self.rules[NT][P]
                else:
                    rules[NT][P] = self.rules[NT][P]
        # Cleaning produces infinite loop
        return self.__class__(self.start, rules, clean=False)

    @classmethod
    def size_constraint(
        cls, dsl: DSL, type_request: Type, max_size: int, n_gram: int = 2
    ) -> "TTCFG[NGram, Tuple[int, int]]":
        """
        Constructs a n-gram TT CFG from a DSL imposing the maximum program size.

        max_size: int - is the maxium depth of programs allowed
        """
        forbidden_sets = dsl.forbidden_patterns

        def __transition__(
            state: Tuple[Type, Tuple[NGram, Tuple[int, int]]],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, Tuple[int, int]]:
            predecessors = state[1][0]
            last_pred = predecessors.last() if len(predecessors) > 0 else None
            if derivation in forbidden_sets.get(
                (last_pred[0].primitive, last_pred[1])
                if last_pred and isinstance(last_pred[0], Primitive)
                else ("", 0),
                set(),
            ):
                return False, (0, 0)
            size, future = state[1][1]
            if size > max_size:
                return False, (0, 0)
            if not derivation.type.is_instance(Arrow):
                if future > 0:
                    return size + future <= max_size, (size + 1, future - 1)
                return size + 1 + future <= max_size, (size + 1, future)
            nargs = len(derivation.type.arguments())
            if future > 0:
                return size + nargs + future <= max_size, (size + 1, future + nargs - 1)
            return size + nargs + 1 + future <= max_size, (size + 1, future + nargs)

        return __saturation_build__(
            dsl,
            type_request,
            (NGram(n_gram), (0, 0)),
            __transition__,
            lambda ctx, P, i, __: ctx[1][0].successor((P, i)),
        )

    @classmethod
    def at_most_k(
        cls, dsl: DSL, type_request: Type, primitive: str, k: int, n_gram: int = 2
    ) -> "TTCFG[NGram, int]":
        """
        Constructs a n-gram TT CFG from a DSL imposing at most k occurences of a certain primitive.
        """
        forbidden_sets = dsl.forbidden_patterns

        def __transition__(
            state: Tuple[Type, Tuple[NGram, int]],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            predecessors = state[1][0]
            last_pred = predecessors.last() if len(predecessors) > 0 else None
            if derivation in forbidden_sets.get(
                (last_pred[0].primitive, last_pred[1])
                if last_pred and isinstance(last_pred[0], Primitive)
                else ("", 0),
                set(),
            ):
                return False, 0
            occ_left = state[1][1]
            if str(derivation) != primitive:
                return True, occ_left
            return occ_left > 0, occ_left - 1

        return __saturation_build__(
            dsl,
            type_request,
            (NGram(n_gram), k),
            __transition__,
            lambda ctx, P, i, __: ctx[1][0].successor((P, i)),
        )


def __saturation_build__(
    dsl: DSL,
    type_request: Type,
    init: Tuple[S, T],
    transition: Callable[
        [Tuple[Type, Tuple[S, T]], Union[Primitive, Variable, Constant]], Tuple[bool, T]
    ],
    get_non_terminal: Callable[
        [Tuple[Type, Tuple[S, T]], Union[Primitive, Variable, Constant], int, Type], S
    ],
) -> "TTCFG[S, T]":
    """
    Abstract builder of TTCFG
    """
    rules: Dict[
        Tuple[Type, Tuple[S, T]],
        Dict[DerivableProgram, Tuple[List[Tuple[Type, S]], T]],
    ] = {}

    return_type = type_request.returns()
    args = type_request.arguments()

    list_to_be_treated: Deque[Tuple[Tuple[Type, S], T, List[Tuple[Type, S]]]] = deque()
    list_to_be_treated.append(((return_type, init[0]), init[1], []))

    while list_to_be_treated:
        (current_type, non_terminal), current, stack = list_to_be_treated.pop()
        rule = current_type, (non_terminal, current)
        # Create rule if non existent
        if rule not in rules:
            rules[rule] = {}
        else:
            continue
        # Try to add variables rules
        for i in range(len(args)):
            if current_type == args[i]:
                var = Variable(i, current_type)
                can_add, new_el = transition(rule, var)
                if can_add:
                    rules[rule][var] = ([], new_el)
                    if stack:
                        list_to_be_treated.append((stack[0], new_el, stack[1:]))
        # DSL Primitives
        for P in dsl.list_primitives:
            type_P = P.type
            arguments_P = type_P.ends_with(current_type)
            if arguments_P is not None:
                can_add, new_el = transition(rule, P)
                if can_add:
                    decorated_arguments_P = [
                        (arg, get_non_terminal(rule, P, i, arg))
                        for i, arg in enumerate(arguments_P)
                    ]
                    rules[rule][P] = (decorated_arguments_P, new_el)
                    tmp_stack = decorated_arguments_P + stack
                    if tmp_stack:
                        list_to_be_treated.append((tmp_stack[0], new_el, tmp_stack[1:]))

    return TTCFG((return_type, init), rules)
