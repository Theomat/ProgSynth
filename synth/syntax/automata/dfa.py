from typing import Callable, Dict, Generic, Set, Tuple, TypeVar


U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")


class DFA(Generic[U, V]):
    """
    Deterministic safe finite automaton.
    Reads V elements from states U.
    If there is no transition from U reading V it means it is non accepting. (there are no final states)
    """

    def __init__(self, initial: U, rules: Dict[U, Dict[V, U]], loops: Set[U]) -> None:
        self.start = initial
        self.rules = rules
        self.loops = loops

    def __mul__(self, other: "DFA[W, X]") -> "DFA[Tuple[U, W], Tuple[V, X]]":
        start = (self.start, other.start)
        rules: Dict[Tuple[U, W], Dict[Tuple[V, X], Tuple[U, W]]] = {}
        new_loops = set()
        for S1 in self.rules:
            for S2 in other.rules:
                if S1 in self.loops and S2 in other.loops:
                    new_loops.add((S1, S2))
                rules[(S1, S2)] = {}
                for w1 in self.rules[S1]:
                    for w2 in other.rules[S2]:
                        rules[(S1, S2)][(w1, w2)] = (
                            self.rules[S1][w1],
                            other.rules[S2][w2],
                        )
        return DFA(start, rules, new_loops)

    @property
    def states(self) -> Set[U]:
        all = set()
        last_frontier = [self.start]
        while last_frontier:
            new_frontier = []
            while last_frontier:
                state = last_frontier.pop()
                for P in self.rules[state]:
                    new_state = self.rules[state][P]
                    if new_state not in all:
                        all.add(new_state)
                        new_frontier.append(new_state)
            last_frontier = new_frontier
        return all

    def can_read(self, start: U, word: V) -> bool:
        return start in self.rules and word in self.rules[start] or start in self.loops

    def read(self, start: U, word: V) -> U:
        if start in self.loops:
            return start
        return self.rules[start][word]

    def map_states(self, f: Callable[[U], W]) -> "DFA[W, V]":
        mapping = {s: f(s) for s in self.states}
        dst_rules = {
            mapping[S]: {P: mapping[self.rules[S][P]] for P in self.rules[S]}
            for S in self.rules
        }
        return DFA(mapping[self.start], dst_rules, {mapping[f] for f in self.loops})

    def then(self, other: "DFA[U, V]") -> "DFA[U, V]":
        assert self.states.isdisjoint(other.states)
        new_rules = {
            S: {
                P: self.rules[S][P]
                if self.rules[S][P] not in self.loops
                else other.start
                for P in self.rules[S]
            }
            for S in self.rules
        }
        for S in other.rules:
            new_rules[S] = {P: other.rules[S][P] for P in other.rules[S]}
        return DFA(self.start, new_rules, other.loops)

    def read_product(self, other: "DFA[W, V]") -> "DFA[Tuple[U, W], V]":
        start = (self.start, other.start)
        rules: Dict[Tuple[U, W], Dict[V, Tuple[U, W]]] = {}
        new_loops = set()
        for S1 in self.rules:
            for S2 in other.rules:
                if S1 in self.loops and S2 in other.loops:
                    new_loops.add((S1, S2))
                rules[(S1, S2)] = {}
                for v in self.rules[S1]:
                    if v in other.rules[S2]:
                        rules[(S1, S2)][v] = (
                            self.rules[S1][v],
                            other.rules[S2][v],
                        )
        return DFA(start, rules, new_loops)
