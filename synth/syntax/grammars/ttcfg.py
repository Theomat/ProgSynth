from collections import deque
from typing import (
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
            if isinstance(other.rules[other.start], tuple):
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
        new_rules: Dict[Tuple[Type, Tuple[S, T]], Set[DerivableProgram]] = {}
        list_to_be_treated: Deque[
            Tuple[Tuple[Type, S], T, List[Tuple[Type, S]]]
        ] = deque()
        list_to_be_treated.append(
            ((self.start[0], self.start[1][0]), self.start[1][1], [])
        )

        if isinstance(self.type_request, Arrow):
            args = self.type_request.arguments()
        else:
            args = []

        while list_to_be_treated:
            (current_type, non_terminal), current, stack = list_to_be_treated.pop()
            rule = current_type, (non_terminal, current)
            # Create rule if non existent
            if rule not in new_rules:
                new_rules[rule] = set()
            else:
                continue
            # Try to add variables rules
            for i in range(len(args)):
                if current_type == args[i]:
                    var = Variable(i, current_type)
                    if var in self.rules[rule]:
                        new_rules[rule].add(var)
                        _, s = self.rules[rule][var]
                        if stack:
                            list_to_be_treated.append((stack[0], s, stack[1:]))
            # DSL Primitives
            for P in self.rules[rule]:
                type_P = P.type
                arguments_P = type_P.ends_with(current_type)
                if arguments_P is not None:
                    if P in self.rules[rule]:
                        decorated_arguments_P, new_el = self.rules[rule][P]
                        new_rules[rule].add(P)
                        tmp_stack = decorated_arguments_P + stack
                        if tmp_stack:
                            list_to_be_treated.append(
                                (tmp_stack[0], new_el, tmp_stack[1:])
                            )

        self.rules = {S: {P: self.rules[S][P] for P in new_rules[S]} for S in new_rules}

        # Now back propagate and delete things that cannot finish with a valid program
        deleted_S = []
        # Fill deleted S
        for S in list(self.rules.keys()):
            if len(self.rules[S]) == 0:
                deleted_S.append(S)
                del self.rules[S]
        while deleted_S:
            S = deleted_S.pop()
            to_check = (S[0], S[1][0])
            for SS in list(self.rules.keys()):
                for P in list(self.rules[SS].keys()):
                    derlist, state = self.rules[SS][P]
                    to_delete = False
                    for der in derlist:
                        if der == to_check:
                            to_delete = True
                            break
                    if to_delete:
                        del self.rules[SS][P]
                        if len(self.rules[SS]) == 0:
                            del self.rules[SS]
                            deleted_S.append(SS)

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

    def possible_outcomes_after(
        self,
        S: Tuple[Type, Tuple[S, T]],
        P: Optional[DerivableProgram],
        info: Optional[List[Tuple[Type, S]]],
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

    @classmethod
    def size_constraint(
        cls, dsl: DSL, type_request: Type, max_size: int, n_gram: int = 2
    ) -> "TTCFG[NGram, Tuple[int, int]]":
        """
        Constructs a n-gram TT CFG from a DSL imposing the maximum program size.

        max_size: int - is the maxium depth of programs allowed
        """
        dsl.instantiate_forbidden()
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
            if not isinstance(derivation.type, Arrow):
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
        dsl.instantiate_forbidden()
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

    if isinstance(type_request, Arrow):
        return_type = type_request.returns()
        args = type_request.arguments()
    else:
        return_type = type_request
        args = []

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

    return TTCFG((return_type, init), rules, clean=False)
