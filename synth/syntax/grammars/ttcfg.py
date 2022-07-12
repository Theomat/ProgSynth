from collections import deque
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Tuple,
    TypeVar,
    Generic,
    Union,
)

from synth.syntax.dsl import DSL
from synth.syntax.program import Constant, Primitive, Variable
from synth.syntax.type_system import Arrow, Type, UnknownType
from synth.syntax.grammars.det_grammar import DerivableProgram, DetGrammar

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

    def __mul__(self, other: "TTCFG[U, V]") -> "TTCFG[Tuple[S, U], Tuple[T, V]]":
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

    def _remove_non_reachable_(self) -> None:
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable = set([self.start])

        # Compute the list of reachable
        list_to_be_treated: Deque[
            Tuple[Tuple[Type, Tuple[S, T]], List[Tuple[Type, S]]]
        ] = deque()
        list_to_be_treated.append((self.start, []))
        while list_to_be_treated:
            rule, stack = list_to_be_treated.pop()
            if rule not in self.rules:
                continue
            for P in self.rules[rule]:
                args, state = self.rules[rule][P]
                if args:
                    nstack = args + stack
                    nrule = (nstack[0][0], (nstack[0][1], state))
                    if nrule not in reachable:
                        reachable.add(nrule)
                        list_to_be_treated.append((nrule, nstack[1:]))
                elif stack:
                    nrule = (stack[0][0], (stack[0][1], state))
                    if nrule not in reachable:
                        reachable.add(nrule)
                        list_to_be_treated.append((nrule, stack[1:]))

        for rule in list(self.rules.keys()):
            if rule not in reachable:
                del self.rules[rule]

    def _remove_non_producible_(self) -> None:
        pass

    def _start_information_(self) -> List[Tuple[Type, S]]:
        return []

    def __rule_to_str__(
        self, P: DerivableProgram, out: Tuple[List[Tuple[Type, S]], T]
    ) -> str:
        args, state = out
        return "{}: {}\t\t{}".format(P, state, args)

    def name(self) -> str:
        return "TTCFG"

    @classmethod
    def depth_constraint(
        cls,
        dsl: DSL,
        type_request: Type,
        max_depth: int,
        min_variable_depth: int = 1,
    ) -> "TTCFG[str, int]":
        """
        Constructs a TT CFG from a DSL imposing the maximum program depth.

        max_depth: int - is the maxium depth of programs allowed
        min_variable_depth: int - min depth at which variables and constants are allowed
        """

        def __transition__(
            state: Tuple[Type, Tuple[str, int]],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            depth = state[1][1]
            if depth > max_depth:
                return False, 0
            if not isinstance(derivation.type, Arrow):
                if isinstance(derivation, (Variable, Constant)):
                    return depth >= min_variable_depth, depth
                return True, depth
            return depth + 1 <= max_depth, depth + 1

        return __saturation_build__(
            dsl,
            type_request,
            ("start", 1),
            __transition__,
            lambda _, P, i, __: f"{P} arg n°{i}",
        )

    @classmethod
    def size_constraint(
        cls,
        dsl: DSL,
        type_request: Type,
        max_size: int,
    ) -> "TTCFG[str, int]":
        """
        Constructs a TT CFG from a DSL imposing the maximum program size.

        max_size: int - is the maxium depth of programs allowed
        """

        def __transition__(
            state: Tuple[Type, Tuple[str, int]],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            size = state[1][1]
            if size > max_size:
                return False, 0
            if not isinstance(derivation.type, Arrow):
                return size + 1 <= max_size, size + 1
            return size + 1 + len(derivation.type.arguments()) <= max_size, size + 1

        return __saturation_build__(
            dsl,
            type_request,
            ("start", 0),
            __transition__,
            lambda _, P, i, __: f"{P} arg n°{i}",
        )

    @classmethod
    def at_most_k(
        cls,
        dsl: DSL,
        type_request: Type,
        primitive: str,
        k: int,
    ) -> "TTCFG[str, int]":
        """
        Constructs a TT CFG from a DSL imposing at most k occurences of a certain primitive.
        """

        def __transition__(
            state: Tuple[Type, Tuple[str, int]],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            occ_left = state[1][1]
            if str(derivation) != primitive:
                return True, occ_left
            return occ_left > 0, occ_left - 1

        return __saturation_build__(
            dsl,
            type_request,
            ("start", k),
            __transition__,
            lambda _, P, i, __: f"{P} arg n°{i}",
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

    return TTCFG((return_type, init), rules)
