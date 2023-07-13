from collections import defaultdict
from functools import lru_cache
from typing import (
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

from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.dsl import DSL
from synth.syntax.grammars.cfg import CFG, CFGState
from synth.syntax.grammars.grammar import DerivableProgram, NGram
from synth.syntax.grammars.u_grammar import UGrammar
from synth.syntax.type_system import Type, UnknownType

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")


def __extract__(t: Tuple) -> Tuple:
    while len(t) == 1 and isinstance(t, tuple):
        t = t[0]
    return t


def __d2state__(t: Union[Tuple[Type, U], Tuple[Tuple[Type, U], ...]]) -> Tuple[Type, U]:
    t = __extract__(t)
    if isinstance(t[0], tuple):
        # Get the type
        our_type = t
        while isinstance(our_type, tuple):
            our_type = our_type[0]  # type: ignore
        # Compute the rest
        rest = []
        for tt in t:
            rest.append(__extract__(tt)[1])
        return (our_type, tuple(rest))
    return t  # type: ignore


class UCFG(UGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], Generic[U]):
    """
    Represents an unambigous context-free grammar.

    """

    def __init__(
        self,
        starts: Set[Tuple[Type, U]],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, List[List[Tuple[Type, U]]]]],
        clean: bool = True,
    ):
        self._some_start = list(starts)[0]
        super().__init__(starts, rules, clean)

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
        done: Set[Tuple[Tuple[Tuple[Type, U], ...], Tuple[Type, U]]] = set()
        reached = {x for x in self.starts}
        for x in self.starts:
            done.add((tuple(self.start_information()), x))
        to_test = [(x, self.start_information()) for x in self.starts]

        while to_test:
            S, info = to_test.pop()
            for P in self.rules[S]:
                for a in self.derive(info, S, P):
                    new_info, next_S, _ = a
                    i = len(done)
                    done.add((tuple(new_info), next_S))
                    if len(done) == i:
                        continue
                    reached.add(next_S)
                    if isinstance(next_S[0], UnknownType):
                        continue
                    to_test.append((next_S, new_info))

        self.rules = {
            S: {P: possibles for P, possibles in dicP.items()}
            for S, dicP in self.rules.items()
            if S in reached
        }

        # Clean starts
        next_starts = set()
        for S in self.starts:
            has_one = False
            for P in self.rules[S]:
                for a in self.derive(self.start_information(), S, P):
                    new_info, next_S, _ = a
                    if (tuple(new_info), next_S) in done or isinstance(
                        next_S[0], UnknownType
                    ):
                        has_one = True
                        break
                if has_one:
                    break
            if has_one:
                next_starts.add(S)

        self.starts = next_starts

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

    def derive_specific(
        self,
        information: List[Tuple[Type, U]],
        S: Tuple[Type, U],
        P: DerivableProgram,
        v: List[Tuple[Type, U]],
    ) -> Optional[Tuple[List[Tuple[Type, U]], Tuple[Type, U]]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        if len(v) == 0:
            if information:
                return (information[1:], information[0])
            else:
                # This indicates the end of a derivation
                return ([], (UnknownType(), self._some_start[1]))
        for args in self.rules[S][P]:
            if args == v:
                return (args[1:] + information, args[0])
        return None

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

        Parameters:
        -----------
        - max_depth: the maximum depth of programs allowed
        - upper_bound_size_type: the maximum size type allowed for polymorphic type instanciations
        - min_variable_depth: min depth at which variables and constants are allowed
        - n_gram: the context, a bigram depends only in the parent node
        - recursive: enables the generated programs to call themselves
        - constant_types: the set of of types allowed for constant objects
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
        return UCFG.from_CFG(cfg, True)

    @classmethod
    def from_CFG(cls, cfg: CFG, clean: bool = False) -> "UCFG[CFGState]":
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
        return UCFG({(cfg.start[0], cfg.start[1][0])}, rules, clean)

    @overload
    @classmethod
    def from_DFTA(
        cls, dfta: DFTA[Tuple[Type, U], DerivableProgram], clean: bool = True
    ) -> "UCFG[U]":
        pass

    @overload
    @classmethod
    def from_DFTA(
        cls,
        dfta: DFTA[Tuple[Tuple[Type, U], ...], DerivableProgram],
        clean: bool = True,
    ) -> "UCFG[Tuple[U, ...]]":
        pass

    @classmethod
    def from_DFTA(
        cls,
        dfta: Union[
            DFTA[Tuple[Tuple[Type, U], ...], DerivableProgram],
            DFTA[Tuple[Type, U], DerivableProgram],
        ],
        clean: bool = True,
    ) -> "Union[UCFG[U], UCFG[Tuple[U, ...]]]":
        """
        Convert a DFTA into a UCFG representing the same language.
        """

        starts = {__d2state__(q) for q in dfta.finals}

        new_rules: Dict[
            Tuple[Type, U],
            Dict[DerivableProgram, List[List[Tuple[Type, U]]]],
        ] = {}

        stack: List[Tuple[Type, U]] = [el for el in starts]
        while stack:
            tgt = stack.pop()
            if tgt in new_rules:
                continue
            new_rules[tgt] = defaultdict(list)
            for (P, args), dst in dfta.rules.items():
                if __d2state__(dst) != tgt:
                    continue
                new_rules[tgt][P].append([__d2state__(arg) for arg in args])
                for new_state in args:
                    if __d2state__(new_state) not in new_rules:
                        stack.append(__d2state__(new_state))

        return UCFG(starts, new_rules, clean)

    @classmethod
    def from_DFTA_with_ngrams(
        cls,
        dfta: Union[
            DFTA[Tuple[Tuple[Type, U], ...], DerivableProgram],
            DFTA[Tuple[Type, U], DerivableProgram],
        ],
        ngram: int,
        clean: bool = False,
    ) -> "Union[UCFG[Tuple[NGram, U]], UCFG[Tuple[NGram, Tuple[U, ...]]]]":
        """
        Convert a DFTA into a UCFG representing the same language and adds contextual information with ngrams.
        If the DFTA is reduced then the UCFG is, therefore in that case clean should be set to False since cleaning can be expensive.
        """

        def local_d2state(
            t: Union[Tuple[Type, U], Tuple[Tuple[Type, U], ...]], v: Optional[NGram]
        ) -> Tuple[Type, Tuple[NGram, U]]:
            a, b = __d2state__(t)
            dst_v = v or NGram(ngram)
            return (a, (dst_v, b))

        match = lambda a, b: a[0] == b[0] and a[1][1] == b[1][1]

        starts = {local_d2state(q, None) for q in dfta.finals}

        new_rules: Dict[
            Tuple[Type, Tuple[NGram, U]],
            Dict[DerivableProgram, List[List[Tuple[Type, Tuple[NGram, U]]]]],
        ] = {}

        stack = [el for el in starts]
        while stack:
            tgt = stack.pop()
            if tgt in new_rules:
                continue
            new_rules[tgt] = defaultdict(list)
            last: NGram = tgt[1][0]
            for (P, args), dst in dfta.rules.items():
                if not match(local_d2state(dst, None), tgt):
                    continue
                new_args = [
                    local_d2state(arg, last.successor((P, i))) for i, arg in enumerate(args)  # type: ignore
                ]
                new_rules[tgt][P].append(new_args)
                for new_state in new_args:
                    if new_state not in new_rules:
                        stack.append(new_state)

        return UCFG(starts, new_rules, clean)
