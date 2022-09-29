from collections import defaultdict
from functools import lru_cache
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Union,
    overload,
)
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.dsl import DSL

from synth.syntax.grammars.cfg import CFG, CFGState
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.grammars.u_grammar import UGrammar
from synth.syntax.type_system import Type, UnknownType

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")


class UCFG(UGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], Generic[U]):
    def __init__(
        self,
        starts: Set[Tuple[Type, U]],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, List[List[Tuple[Type, U]]]]],
        clean: bool = True,
    ):
        super().__init__(starts, rules, clean)
        self._some_start = list(self.starts)[0]

    def name(self) -> str:
        return "UCFG"

    def __hash__(self) -> int:
        return super().__hash__()

    def __rule_to_str__(self, P: DerivableProgram, out: List[Tuple[Type, U]]) -> str:
        return "{}: {}".format(P, out)

    def clean(self) -> None:
        """
        Clean this unambiguous grammar by removing non reachable, non productive rules.
        """
        # TODO
        pass

    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        possibles = self.rules[S][P]
        return len(possibles[0])

    def start_information(self) -> List[Tuple[Type, U]]:
        return []

    def derive(
        self, information: List[Tuple[Type, U]], S: Tuple[Type, U], P: DerivableProgram
    ) -> List[Tuple[List[Tuple[Type, U]], Tuple[Type, U], List[Tuple[Type, U]]]]:
        """
        Given the current information and the derivation S -> P, produces the list of possibles new information state and the next S after this derivation.
        """
        if S not in self.rules or P not in self.rules[S]:
            # This is important since we can fail to derive
            return []
        candidates = self.rules[S][P]
        out = []
        for args in candidates:
            if args:
                out.append((args[1:] + information, args[0], args))
            elif information:
                out.append((information[1:], information[0], []))
            else:
                # This indicates the end of a derivation
                out.append(([], (UnknownType(), self._some_start[1]), []))
        return out

    @lru_cache()
    def programs(self) -> int:
        """
        Return the total number of programs contained in this grammar.
        """
        _counts: Dict[Tuple[Type, U], int] = {}

        def __compute__(state: Tuple[Type, U]) -> int:
            if state in _counts:
                return _counts[state]
            if state not in self.rules:
                return 1
            total = 0
            for P in self.rules[state]:
                possibles = self.derive(self.start_information(), state, P)
                for _, _, args in possibles:
                    local = 1
                    for arg in args:
                        local *= __compute__(arg)
                    total += local
            _counts[state] = total
            return total

        return sum(__compute__(start) for start in self.starts)

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
        return UCFG({(cfg.start[0], cfg.start[1][0])}, rules)

    @overload
    @classmethod
    def from_DFTA(cls, dfta: DFTA[Tuple[Type, U], DerivableProgram]) -> "UCFG[U]":
        pass

    @overload
    @classmethod
    def from_DFTA(
        cls, dfta: DFTA[Tuple[Tuple[Type, U], ...], DerivableProgram]
    ) -> "UCFG[Tuple[U, ...]]":
        pass

    @classmethod
    def from_DFTA(
        cls,
        dfta: Union[
            DFTA[Tuple[Tuple[Type, U], ...], DerivableProgram],
            DFTA[Tuple[Type, U], DerivableProgram],
        ],
    ) -> "Union[UCFG[U], UCFG[Tuple[U, ...]]]":
        d2state = lambda x: x
        some_final = list(dfta.finals)[0]
        if isinstance(some_final[0], tuple):
            d2state = lambda t: (t[0][0], tuple(tt[1] for tt in t))

        starts = {d2state(q) for q in dfta.finals}

        new_rules: Dict[
            Tuple[Type, U],
            Dict[DerivableProgram, List[List[Tuple[Type, U]]]],
        ] = {}

        stack = [el for el in starts]
        while stack:
            tgt = stack.pop()
            if tgt in new_rules:
                continue
            new_rules[tgt] = defaultdict(list)
            for (P, args), dst in dfta.rules.items():
                if d2state(dst) != tgt:
                    continue
                new_rules[tgt][P].append([d2state(arg) for arg in args])
                for new_state in args:
                    if d2state(new_state) not in new_rules:
                        stack.append(d2state(new_state))

        return UCFG(starts, new_rules)
