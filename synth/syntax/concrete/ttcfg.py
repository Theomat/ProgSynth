from collections import deque
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Union,
)

from synth.syntax.dsl import DSL
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type

T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")
V = TypeVar("V")


class TTCFG(Generic[S, T]):
    """
    Represents a deterministic Tree Traversing CFG (TTCFG).
    """

    def __init__(
        self,
        start: Tuple[Type, S, T],
        rules: Dict[
            Tuple[Type, S, T],
            Dict[Union[Primitive, Variable, Constant], Tuple[List[Tuple[Type, S]], T]],
        ],
        clean: bool = False,
    ):
        self.start = start
        self.rules = rules

        if clean:
            self.clean()

        # Compute the type request
        type_req = self.start[0]
        variables: List[Variable] = []
        for S in self.rules:
            for P in self.rules[S]:
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

    def clean(self) -> None:
        """
        Remove non reachable
        """
        self.__remove_non_reachable__()

    def __hash__(self) -> int:
        return hash((self.start, str(self.rules)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TTCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __str__(self) -> str:
        s = "Print a TT CFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, state = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P, args_P, state)
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, other: "TTCFG[U, V]") -> "TTCFG[Tuple[S, U], Tuple[T, V]]":
        assert (
            self.type_request == other.type_request
        ), "Both TTCFGs do not have the same type request!"
        rules: Dict[
            Tuple[Type, Tuple[S, U], Tuple[T, V]],
            Dict[
                Union[Primitive, Variable, Constant],
                Tuple[List[Tuple[Type, Tuple[S, U]]], Tuple[T, V]],
            ],
        ] = {}
        start: Tuple[Type, Tuple[S, U], Tuple[T, V]] = (
            self.start[0],
            (self.start[1], other.start[1]),
            (self.start[2], other.start[2]),
        )
        for nT1 in self.rules:
            for nT2 in other.rules:
                # check type equality
                if nT1[0] != nT2[0]:
                    continue
                rule = (nT1[0], (nT1[1], nT2[1]), (nT1[2], nT2[2]))
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

    def __remove_non_reachable__(self) -> None:
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable = set([self.start])

        # Compute the list of reachable
        list_to_be_treated: Deque[
            Tuple[Tuple[Type, S, T], List[Tuple[Type, S]]]
        ] = deque()
        list_to_be_treated.append((self.start, []))
        while list_to_be_treated:
            rule, stack = list_to_be_treated.pop()
            for P in self.rules[rule]:
                args, state = self.rules[rule][P]
                if args:
                    nstack = args + stack
                    if nstack:
                        nrule = (nstack[0][0], nstack[0][1], state)
                        if nrule not in reachable:
                            reachable.add(nrule)
                            list_to_be_treated.append((nrule, nstack[1:]))
                elif stack:
                    nrule = (stack[0][0], stack[0][1], state)
                    if nrule not in reachable:
                        reachable.add(nrule)
                        list_to_be_treated.append((nrule, stack[1:]))

        for rule in list(self.rules.keys()):
            if rule not in reachable:
                del self.rules[rule]

    def __parse__(
        self, P: Program, S: Optional[Tuple[Type, S, T]] = None
    ) -> Optional[T]:
        S = S or self.start
        if S not in self.rules:
            return None
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            assert isinstance(F, Primitive)
            if F not in self.rules[S]:
                return None
            current: Optional[T] = self.rules[S][F][1]
            if current is None:
                return None
            for nc, arg in zip(self.rules[S][F][0], args_P):
                current = self.__parse__(arg, (nc[0], nc[1], current))
                if current is None:
                    return None
            return current

        elif isinstance(P, (Variable, Primitive)):
            return self.rules[S].get(P, (None, None))[1]
        return None

    def __contains__(self, P: Program, S: Optional[Tuple[Type, S, T]] = None) -> bool:
        return self.__parse__(P, S) is not None

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
            state: Tuple[Type, str, int],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            depth = state[2]
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
            state: Tuple[Type, str, int],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            size = state[2]
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
            state: Tuple[Type, str, int],
            derivation: Union[Primitive, Variable, Constant],
        ) -> Tuple[bool, int]:
            occ_left = state[2]
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
        [Tuple[Type, S, T], Union[Primitive, Variable, Constant]], Tuple[bool, T]
    ],
    get_non_terminal: Callable[
        [Tuple[Type, S, T], Union[Primitive, Variable, Constant], int, Type], S
    ],
) -> "TTCFG[S, T]":
    rules: Dict[
        Tuple[Type, S, T],
        Dict[Union[Primitive, Variable, Constant], Tuple[List[Tuple[Type, S]], T]],
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
        rule = current_type, non_terminal, current
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

    return TTCFG((return_type, init[0], init[1]), rules)
