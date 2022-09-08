from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Union,
)
from functools import lru_cache
import copy
from synth.syntax.dsl import are_equivalent_primitives

from synth.syntax.grammars.grammar import Grammar
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")

DerivableProgram = Union[Primitive, Variable, Constant]


class UGrammar(Grammar, ABC, Generic[U, V, W]):
    def __init__(
        self,
        start: Tuple[Type, U],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, List[V]]],
        clean: bool = True,
    ):
        self.start = start
        self.rules = rules
        self.type_request = self._guess_type_request_()
        if clean:
            self.clean()

    @lru_cache()
    def primitives_used(self) -> Set[Primitive]:
        """
        Returns the set of primitives used by this grammar.
        """
        out: Set[Primitive] = set()
        for S in self.rules:
            for P in self.rules[S]:
                if isinstance(P, Primitive):
                    out.add(P)
        return out

    def __hash__(self) -> int:
        return hash((self.start, str(self.rules)))

    def __rule_to_str__(self, P: DerivableProgram, out: V) -> str:
        return "{}: {}".format(P, out)

    def __str__(self) -> str:
        s = f"Print a {self.name()}\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                out = self.rules[S][P]
                for possible in out:
                    s += "   {}\n".format(self.__rule_to_str__(P, possible))
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def _guess_type_request_(self) -> Type:
        """
        Guess the type request of this grammar.
        """
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
        return type_req

    def __contains__(self, program: Program) -> bool:
        return self.__contains_rec__(program, self.start, self.start_information())[0]

    def __contains_rec__(
        self, program: Program, start: Tuple[Type, U], information: W
    ) -> Tuple[bool, List[Tuple[W, Tuple[Type, U]]]]:
        if start not in self.rules:
            return False, [(information, start)]
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            if function not in self.rules[start]:
                return False, [(information, start)]
            possibles = self.derive(information, start, function)  # type: ignore
            for arg in args_P:
                next_possibles = []
                for possible in possibles:
                    information, next = possible
                    contained, new_possibles = self.__contains_rec__(
                        arg, start=next, information=information
                    )
                    if contained:
                        next_possibles += new_possibles
                if len(next_possibles) == 0:
                    return False, [(information, next)]
                possibles = next_possibles
            return True, possibles
        elif isinstance(program, (Primitive, Variable, Constant)):
            if program not in self.rules[start]:
                return False, [(information, start)]
            possibles = self.derive(information, start, program)
            return True, possibles
        return False, [(information, start)]

    def clean(self) -> None:
        """
        Clean this deterministic grammar by removing non reachable, non productive rules.
        """
        pass

    @abstractmethod
    def derive(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram
    ) -> List[Tuple[W, Tuple[Type, U]]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        pass

    def derive_all(
        self,
        information: W,
        S: Tuple[Type, U],
        P: Program,
        current: Optional[List[Tuple[Type, U]]] = None,
    ) -> List[Tuple[W, List[Tuple[Type, U]]]]:
        """
        Given current information and context S, produces the new information and all the contexts the grammar went through to derive program P.
        """
        current = current or []
        if isinstance(P, (Primitive, Variable, Constant)):
            cur_possibles = self.derive(information, S, P)
            out = [(info, current + [ctx]) for info, ctx in cur_possibles]
            return out

        elif isinstance(P, Function):
            F = P.function
            current.append(S)
            # information, _ = self.derive_all(information, S, F, current)
            possibles = self.derive_all(information, S, F, current)
            for arg in P.arguments:
                next_possibles = []
                for possible in possibles:
                    information, crt = possible
                    S = crt[-1]
                    possibles_arg = self.derive_all(information, S, arg, current)
                    for information, next_crt in possibles_arg:
                        next_possibles.append((information, next_crt))
                possibles = next_possibles
                # information, _ = self.derive_all(information, S, arg, current)
                # S = current[-1]
            return possibles
        assert False

    @abstractmethod
    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        pass

    @abstractmethod
    def start_information(self) -> W:
        pass

    # def reduce_derivations(
    #     self,
    #     reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
    #     init: T,
    #     program: Program,
    #     start: Optional[Tuple[Type, U]] = None,
    # ) -> T:
    #     """
    #     Reduce the given program using the given reduce operator.

    #     reduce is called after derivation.
    #     """

    #     return self.__reduce_derivations_rec__(
    #         reduce, init, program, start or self.start, self.start_information()
    #     )[0]

    # def __reduce_derivations_rec__(
    #     self,
    #     reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
    #     init: T,
    #     program: Program,
    #     start: Tuple[Type, U],
    #     information: W,
    # ) -> Tuple[T, W, Tuple[Type, U]]:
    #     value = init
    #     if isinstance(program, Function):
    #         function = program.function
    #         args_P = program.arguments
    #         information, next = self.derive(information, start, function)  # type: ignore
    #         value = reduce(value, start, function, self.rules[start][function])  # type: ignore
    #         for arg in args_P:
    #             value, information, next = self.__reduce_derivations_rec__(
    #                 reduce, value, arg, start=next, information=information
    #             )
    #         return value, information, next
    #     elif isinstance(program, (Primitive, Variable, Constant)):
    #         information, next = self.derive(information, start, program)
    #         value = reduce(value, start, program, self.rules[start][program])
    #         return value, information, next
    #     return value, information, start
