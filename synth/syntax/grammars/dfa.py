from functools import lru_cache
from typing import Callable, Dict, Generic, Set, Tuple, TypeVar


U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")


class DFA(Generic[U, V]):
    """
    Deterministic finite automaton.
    Reads V elements from states U.
    If there is no transition from U reading V it means it is non accepting. (there are no final states)
    """

    def __init__(self, initial: U, rules: Dict[U, Dict[V, U]], finals: Set[U]) -> None:
        self.start = initial
        self.rules = rules
        self.finals = finals

    def __mul__(self, other: "DFA[W, X]") -> "DFA[Tuple[U, W], Tuple[V, X]]":
        start = (self.start, other.start)
        rules: Dict[Tuple[U, W], Dict[Tuple[V, X], Tuple[U, W]]] = {}
        for S1 in self.rules:
            for S2 in other.rules:
                rules[(S1, S2)] = {}
                for w1 in self.rules[S1]:
                    for w2 in other.rules[S2]:
                        rules[(S1, S2)][(w1, w2)] = (
                            self.rules[S1][w1],
                            other.rules[S2][w2],
                        )
        return DFA(start, rules)

    @property
    @lru_cache
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
        return start in self.rules and word in self.rules[start] or start in self.finals

    def read(self, start: U, word: V) -> U:
        if start in self.finals:
            return start
        return self.rules[start][word]

    def map_states(self, f: Callable[[U], W]) -> "DFA[W, V]":
        mapping = {s: f(s) for s in self.states}
        dst_rules = {
            {mapping[self.rules[S][P]] for P in self.rules[S]} for S in self.rules
        }
        return DFA(mapping[self.start], dst_rules, {mapping[f] for f in self.finals})

    def then(self, other: "DFA[U, V]") -> "DFA[U, V]":
        assert self.states.isdisjoint(other.states)
        new_rules = {
            {
                self.rules[S][P] if self.rules[S][P] not in self.finals else other.start
                for P in self.rules[S]
            }
            for S in self.rules
        }
        for S in other.rules:
            new_rules[S] = {other.rules[S][P] for P in other.rules[S]}
        return DFA(self.start, new_rules, other.finals)
