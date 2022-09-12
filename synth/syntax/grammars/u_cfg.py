from typing import (
    Dict,
    List,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Union,
)
from synth.syntax.dsl import DSL

from synth.syntax.grammars.cfg import CFG, CFGState, NoneType
from synth.syntax.grammars.u_grammar import UGrammar
from synth.syntax.program import Constant, Primitive, Variable
from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")

DerivableProgram = Union[Primitive, Variable, Constant]


class UCFG(UGrammar[U, List[Tuple[Type, U]], NoneType], Generic[U]):
    def name(self) -> str:
        return "UCFG"

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
        if S not in self.rules or P not in self.rules[S]:
            return [(None, S, [])]
        possibles = self.rules[S][P]
        return [
            (None, possible[0] if possible else S, possible) for possible in possibles
        ]

    @classmethod
    def depth_constraint(
        cls,
        dsl: DSL,
        type_request: Type,
        max_depth: int,
        upper_bound_type_size: int = 10,
        min_variable_depth: int = 1,
        n_gram: int = 2,
        recursive: bool = False,
        constant_types: Set[Type] = set(),
    ) -> "UCFG[CFGState]":
        """
        Constructs a UCFG from a DSL imposing bounds on size of the types
        and on the maximum program depth.

        max_depth: int - is the maxium depth of programs allowed
        uppder_bound_size_type: int - is the maximum size type allowed for polymorphic type instanciations
        min_variable_depth: int - min depth at which variables and constants are allowed
        n_gram: int - the context, a bigram depends only in the parent node
        recursvie: bool - allows the generated programs to call themselves
        constant_types: Set[Type] - the set of of types allowed for constant objects
        """
        cfg = CFG.depth_constraint(
            dsl,
            type_request,
            max_depth,
            upper_bound_type_size,
            min_variable_depth,
            n_gram,
            recursive,
            constant_types,
        )
        return UCFG.from_CFG(cfg)

    @classmethod
    def from_CFG(cls, cfg: CFG) -> "UCFG[CFGState]":
        """
        Constructs a UCFG from the specified CFG
        """
        rules: Dict[
            Tuple[Type, CFGState],
            Dict[DerivableProgram, List[List[Tuple[Type, CFGState]]]],
        ] = {}
        for S in cfg.rules:
            nS = (S[0], S[1][0])
            rules[nS] = {}
            for P in cfg.rules[S]:
                rules[nS][P] = [[SS for SS in cfg.rules[S][P][0]]]
        return UCFG((cfg.start[0], cfg.start[1][0]), rules)
