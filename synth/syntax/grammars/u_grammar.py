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

from synth.syntax.grammars.grammar import DerivableProgram, Grammar
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")


class UGrammar(Grammar, ABC, Generic[U, V, W]):
    """
    Represents an unambiguous grammar.

    (S) Non-terminals are Tuple[Type, U].
    (f) are Derivable programs
    derivations are:
    S -> f  | S1  ... Sk
            | S'1 ... S'k
    where both cases are valid derivations.
    S1 ... Sk, S'1 ... S'k are of type V.

    When deriving an information of type W is maintained.

    Parameters:
    -----------
    - starts: the starts non terminal of the grammar
    - rules: the derivation rules

    """

    def __init__(
        self,
        starts: Set[Tuple[Type, U]],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, List[V]]],
        clean: bool = True,
    ):
        self.starts = starts
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
        return hash((tuple(self.starts), str(self.rules)))

    def __rule_to_str__(self, P: DerivableProgram, out: V) -> str:
        return "{}: {}".format(P, out)

    def __str__(self) -> str:
        s = f"Print a {self.name()}\n"
        s += "starts: {}\n".format(self.starts)
        for S in reversed(self.rules):
            add = " [START]" if S in self.starts else ""
            s += "#\n {}{}\n".format(S, add)
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
        type_req = list(self.starts)[0][0]
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
        return any(
            self.__contains_rec__(program, start, self.start_information())[0]
            for start in self.starts
        )

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
            possibles = [(a, b) for a, b, _ in self.derive(information, start, function)]  # type: ignore
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
            possibles = [(a, b) for a, b, _ in self.derive(information, start, program)]
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
    ) -> List[Tuple[W, Tuple[Type, U], V]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        pass

    def derive_specific(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram, v: V
    ) -> Optional[Tuple[W, Tuple[Type, U]]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        for a, b, c in self.derive(information, S, P):
            if c == v:
                return a, b
        return None

    def derive_all(
        self,
        information: W,
        S: Tuple[Type, U],
        P: Program,
        current: Optional[List[Tuple[Tuple[Type, U], V]]] = None,
        hints: Optional[Dict[Tuple[Type, U], Dict[Program, V]]] = None,
    ) -> List[Tuple[W, List[Tuple[Tuple[Type, U], V]]]]:
        """
        Given current information and context S, produces the new information and all the contexts the grammar went through to derive program P.
        """
        current = current or []
        if isinstance(P, (Primitive, Variable, Constant)):
            if hints:
                out_der = self.derive_specific(information, S, P, hints[S][P])
                assert out_der is not None
                return [(out_der[0], current + [(out_der[1], hints[S][P])])]
            else:
                cur_possibles = self.derive(information, S, P)
                out = [(info, current + [(ctx, pp)]) for info, ctx, pp in cur_possibles]
                return out

        elif isinstance(P, Function):
            F = P.function
            # current.append(S)
            # information, _ = self.derive_all(information, S, F, current)
            if hints:
                out_der = self.derive_specific(information, S, F, hints[S][P])  # type: ignore
                assert out_der is not None
                possibles = [(out_der[0], current + [(out_der[1], hints[S][P])])]
                for arg in P.arguments:
                    next_possibles = []
                    for possible in possibles:
                        information, crt = possible
                        S = crt[-1][0]
                        possibles_arg = self.derive_all(
                            information, S, arg, current, hints
                        )
                        for information, next_crt in possibles_arg:
                            next_possibles.append((information, next_crt))
                    possibles = next_possibles
                return possibles
            else:
                possibles = self.derive_all(information, S, F, current)
                for arg in P.arguments:
                    next_possibles = []
                    for possible in possibles:
                        information, crt = possible
                        S = crt[-1][0]
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
    ) -> List[T]:
        """
        Reduce the given program using the given reduce operator.

        reduce: 'a, S, P, V -> 'a

        reduce is called after derivation.
        """

        outputs = []
        if start is None:
            alternatives: List[
                List[Tuple[Tuple[Type, U], Tuple[Type, U], DerivableProgram, V, W]]
            ] = []
            for start in self.starts:
                alternatives += self.__reduce_derivations_rec__(
                    reduce, program, start, self.start_information()
                )
        else:
            alternatives = self.__reduce_derivations_rec__(
                reduce, program, start, self.start_information()
            )
        for possibles in alternatives:
            value = init
            for __, S, P, v, _ in possibles:
                value = reduce(value, S, P, v)
            outputs.append(value)
        return outputs

    def __reduce_derivations_rec__(
        self,
        reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
        program: Program,
        start: Tuple[Type, U],
        information: W,
    ) -> List[List[Tuple[Tuple[Type, U], Tuple[Type, U], DerivableProgram, V, W]]]:
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            possibles: List[List[Tuple[Tuple[Type, U], Tuple[Type, U], DerivableProgram, V, W]]] = [[(b, start, function, c, a)] for a, b, c in self.derive(information, start, function)]  # type: ignore
            for arg in args_P:
                next_possibles = []
                for possible in possibles:
                    next, _, __, v, information = possible[-1]
                    alternatives = self.__reduce_derivations_rec__(
                        reduce, arg, start=next, information=information
                    )
                    for alternative in alternatives:
                        next_possibles.append(possible + alternative)
                possibles = next_possibles
                # value, information, next = self.__reduce_derivations_rec__(
                #     reduce, value, arg, start=next, information=information
                # )
            return next_possibles
        elif isinstance(program, (Primitive, Variable, Constant)):
            new_possibles = self.derive(information, start, program)
            alternatives = []
            for new_possible in new_possibles:
                information, next, v = new_possible
                alternatives.append([(next, start, program, v, information)])
            return alternatives
        return []
