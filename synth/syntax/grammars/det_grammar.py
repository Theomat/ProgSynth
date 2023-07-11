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
)
from functools import lru_cache
import copy

from synth.syntax.grammars.grammar import DerivableProgram, Grammar
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")


class DetGrammar(Grammar, ABC, Generic[U, V, W]):
    """
    Represents a deterministic grammar.

    (S) Non-terminals are Tuple[Type, U].
    (f) are Derivable programs
    derivations are:
    S -> f S1  ... Sk
    there is no other derivation from S using f.
    S1 ... Sk is of type V.

    When deriving an information of type W is maintained.

    Parameters:
    -----------
    - start: the starting non-terminal of the grammar
    - rules: the derivation rules

    """

    def __init__(
        self,
        start: Tuple[Type, U],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, V]],
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
                s += "   {}\n".format(self.__rule_to_str__(P, out))
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def _guess_type_request_(self) -> Type:
        """
        Guess the type request of this grammar.
        """
        # Compute the type request
        type_req = self.start[0]
        self._variables: List[Variable] = []
        for S in self.rules:
            for P in self.rules[S]:
                if isinstance(P, Variable):
                    if P not in self._variables:
                        self._variables.append(P)
        n = len(self._variables)
        for i in range(n):
            j = n - i - 1
            for v in self._variables:
                if v.variable == j:
                    type_req = Arrow(v.type, type_req)
        return type_req

    def variables(self) -> List[Variable]:
        return self._variables[:]

    def __contains__(self, program: Program) -> bool:
        return self.__contains_rec__(program, self.start, self.start_information())[0]

    def __contains_rec__(
        self, program: Program, start: Tuple[Type, U], information: W
    ) -> Tuple[bool, W, Tuple[Type, U]]:
        if start not in self.rules:
            return False, information, start
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            if function not in self.rules[start]:
                return False, information, start
            information, next = self.derive(information, start, function)  # type: ignore
            for arg in args_P:
                contained, information, next = self.__contains_rec__(
                    arg, start=next, information=information
                )
                if not contained:
                    return False, information, next
            return True, information, next
        elif isinstance(program, (Primitive, Variable, Constant)):
            if program not in self.rules[start]:
                return False, information, start
            information, next = self.derive(information, start, program)
            return True, information, next
        return False, information, start

    def clean(self) -> None:
        """
        Clean this deterministic grammar by removing non reachable, non productive rules.
        """
        pass

    @abstractmethod
    def derive(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram
    ) -> Tuple[W, Tuple[Type, U]]:
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
    ) -> Tuple[W, List[Tuple[Type, U]]]:
        """
        Given current information and context S, produces the new information and all the contexts the grammar went through to derive program P.
        """
        current = current or []
        if isinstance(P, (Primitive, Variable, Constant)):
            information, ctx = self.derive(information, S, P)
            current.append(ctx)
            return (information, current)

        elif isinstance(P, Function):
            F = P.function
            current.append(S)
            information, _ = self.derive_all(information, S, F, current)
            S = current[-1]
            for arg in P.arguments:
                information, _ = self.derive_all(information, S, arg, current)
                S = current[-1]
            return (information, current)
        assert False

    @abstractmethod
    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        """
        Returns the number of arguments when deriving P from S.
        """
        pass

    @abstractmethod
    def start_information(self) -> W:
        """
        The starting information when deriving from a starting non-terminal.
        """
        pass

    def reduce_derivations(
        self,
        reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
        init: T,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> T:
        """
        Reduce the given program using the given reduce operator.

        reduce: 'a, S, P, V -> 'a

        reduce is called after derivation.
        """

        return self.__reduce_derivations_rec__(
            reduce, init, program, start or self.start, self.start_information()
        )[0]

    def __reduce_derivations_rec__(
        self,
        reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
        init: T,
        program: Program,
        start: Tuple[Type, U],
        information: W,
    ) -> Tuple[T, W, Tuple[Type, U]]:
        value = init
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            information, next = self.derive(information, start, function)  # type: ignore
            value = reduce(value, start, function, self.rules[start][function])  # type: ignore
            for arg in args_P:
                value, information, next = self.__reduce_derivations_rec__(
                    reduce, value, arg, start=next, information=information
                )
            return value, information, next
        elif isinstance(program, (Primitive, Variable, Constant)):
            information, next = self.derive(information, start, program)
            value = reduce(value, start, program, self.rules[start][program])
            return value, information, next
        return value, information, start

    def embed(self, program: Program) -> Optional[Program]:
        """
        If the DSL has equivalent primitives, try to embed a program without equivalent primtives into this grammar.
        """
        p = self.__embed__(program, self.start, self.start_information())
        return p

    def __embed__(
        self, program: Program, start: Tuple[Type, U], information: W, level: int = 0
    ) -> Optional[Program]:
        if isinstance(program, Function):
            assert isinstance(program.function, Primitive)
            possible_choices = [
                P
                for P in self.rules[start]
                if isinstance(P, Primitive)
                and P.primitive == program.function.primitive
            ]
            for function in possible_choices:
                args_P = program.arguments
                new_information, next = self.derive(
                    copy.deepcopy(information), start, function
                )
                nargs = []
                for arg in args_P:
                    narg = self.__embed__(arg, next, new_information, level + 1)
                    if narg is None:
                        break
                    nargs.append(narg)
                    new_information, lst = self.derive_all(new_information, next, narg)
                    next = lst[-1]
                if len(nargs) != len(args_P):
                    continue
                return Function(function, nargs)
            return None
        elif isinstance(program, Primitive):
            possible_choices = [
                P
                for P in self.rules[start]
                if isinstance(P, Primitive) and P.primitive == program.primitive
            ]
            if len(possible_choices) == 0:
                return None
            elif len(possible_choices) == 1:
                return possible_choices[0]
            assert False, f"Ambigous possibilities: {possible_choices} from {start}"
        else:
            return program if program in self.rules[start] else None
