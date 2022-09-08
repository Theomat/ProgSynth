from typing import (
    List,
    Tuple,
    TypeVar,
    Generic,
    Union,
)

from synth.syntax.grammars.cfg import NoneType
from synth.syntax.grammars.unambigous_grammar import UGrammar
from synth.syntax.program import Constant, Primitive, Variable
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")

DerivableProgram = Union[Primitive, Variable, Constant]


class UCFG(UGrammar[U, List[Tuple[Type, U]], NoneType], Generic[U]):
    def __rule_to_str__(self, P: DerivableProgram, out: List[Tuple[Type, U]]) -> str:
        return "{}: {}".format(P, out)

    def clean(self) -> None:
        """
        Clean this deterministic grammar by removing non reachable, non productive rules.
        """
        pass

    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        possibles = self.rules[S][P]
        return len(possibles[0])

    def start_information(self) -> NoneType:
        return None

    def derive(
        self, information: NoneType, S: Tuple[Type, U], P: DerivableProgram
    ) -> List[Tuple[NoneType, Tuple[Type, U], List[Tuple[Type, U]]]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        possibles = self.rules[S][P]
        return [(None, possible[0], possible) for possible in possibles]
