from typing import Dict, Generic, Set, Tuple, TypeVar

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")


class DFTA(Generic[U, V]):
    """
    Deterministic finite tree automaton.
    states: U
    alphabet: V
    Reads V elements from states U.
    If there is no transition from U reading V it means it is non accepting. (there are no final states)
    """

    def __init__(
        self,
        rules: Dict[
            Tuple[
                V,
                Tuple[U, ...],
            ],
            U,
        ],
        finals: Set[U],
    ) -> None:
        self.finals = finals
        self.rules = rules

    def __mul__(self, other: "DFTA[W, X]") -> "DFTA[Tuple[U, W], Tuple[V, X]]":
        finals = set()
        rules: Dict[
            Tuple[
                Tuple[V, X],
                Tuple[Tuple[U, W], ...],
            ],
            Tuple[U, W],
        ] = {}
        for (l1, args1), dst1 in self.rules.items():
            for (l2, args2), dst2 in other.rules.items():
                if len(args1) != len(args2):
                    continue
                S = ((l1, l2), tuple(zip(args1, args2)))
                if (l1, args1) in self.finals and (l2, args2) in other.finals:
                    finals.add((dst1, dst2))
                rules[S] = (dst1, dst2)
        return DFTA(rules, finals)

    @property
    def states(self) -> Set[U]:
        """
        The set of accessible states.
        """
        reachable = set()
        added = True
        while added:
            added = False
            for (_, args), dst in self.rules.items():
                if dst not in reachable and all(s in reachable for s in args):
                    reachable.add(dst)
                    added = True
        return reachable

    def reduce(self) -> "DFTA[U, V]":
        """
        Return the same DFTA with only accessible states.
        """
        new_states = self.states
        new_rules = {
            (letter, args): dst
            for (letter, args), dst in self.rules.items()
            if dst in new_states and all(s in new_states for s in args)
        }
        return DFTA(new_rules, self.finals.intersection(new_states))

    def read_product(self, other: "DFTA[W, V]") -> "DFTA[Tuple[U, W], V]":
        rules: Dict[
            Tuple[
                V,
                Tuple[Tuple[U, W], ...],
            ],
            Tuple[U, W],
        ] = {}
        finals = set()
        for (l1, args1), dst1 in self.rules.items():
            for (l2, args2), dst2 in other.rules.items():
                if len(args1) != len(args2) or l1 != l2:
                    continue
                S = (l1, tuple(zip(args1, args2)))
                if (l1, args1) in self.finals and (l2, args2) in other.finals:
                    finals.add((dst1, dst2))
                rules[S] = (dst1, dst2)
        return DFTA(rules, finals)
