from typing import Dict, Generic, List, Set, Tuple, TypeVar

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")


class DFTA(Generic[U, V]):
    """
    Deterministic finite tree automaton.
    states: U
    alphabet: V
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
        The set of reachable states.
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
        Return the same DFTA with only reachable states.
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

    def minimise(self) -> "DFTA[Set[U], V]":
        """
        Assumes this is a reduced DTFA

        Adapted algorithm from:
        Brainerd, Walter S.. “The Minimalization of Tree Automata.” Inf. Control. 13 (1968): 484-491.
        """
        # 1. Build relevant
        relevant: Dict[
            U,
            List[
                Tuple[
                    Tuple[
                        V,
                        Tuple[U, ...],
                    ],
                    int,
                ]
            ],
        ] = {q: [] for q in self.states}
        for (l, args) in self.rules:
            for k, ik in enumerate(args):
                relevant[ik].append(((l, args), k))
        # 2. Init equiv classes
        state2cls: Dict[U, int] = {
            q: 0 if q not in self.finals else 1 for q in self.states
        }
        cls2states: Dict[int, Set[U]] = {
            j: {q for q, i in state2cls.items() if i == j} for j in [0, 1]
        }

        n = 1
        finished = False
        # Routine
        def are_equivalent(a: U, b: U) -> bool:
            for S, k in relevant[a]:
                dst_cls = state2cls[self.rules[S]]
                newS = (S[0], tuple([p if j != k else b for j, p in enumerate(S[1])]))
                out = self.rules.get(newS)
                if out is None or state2cls[out] != dst_cls:
                    return False
            for S, k in relevant[b]:
                dst_cls = state2cls[self.rules[S]]
                newS = (S[0], tuple([p if j != k else a for j, p in enumerate(S[1])]))
                out = self.rules.get(newS)
                if out is None or state2cls[out] != dst_cls:
                    return False
            return True

        # 3. Main loop
        while not finished:
            finished = True
            # For each equivalence class
            for i in range(n):
                cls = list(cls2states[i])
                new_cls = []
                while cls:
                    representative = cls.pop()
                    new_cls.append(representative)
                    for q in cls:
                        if are_equivalent(representative, q):
                            new_cls.append(q)
                    cls = [q for q in cls if q not in new_cls]
                    if len(cls) != 0:
                        # Create new equivalence class
                        n += 1
                        for q in new_cls:
                            state2cls[q] = n
                        cls2states[n] = set(new_cls)
                        finished = False
                    else:
                        # If cls is empty then we can use the previous number- that is i- for the new equivalence class
                        cls2states[i] = set(new_cls)

        new_rules: Dict[
            Tuple[
                V,
                Tuple[Set[U], ...],
            ],
            Set[U],
        ] = {}
        for (l, args), dst in self.rules.items():
            new_rules[
                (l, tuple([cls2states[state2cls[q]] for q in args]))
            ] = cls2states[state2cls[dst]]
        return DFTA(new_rules, {cls2states[state2cls[q]] for q in self.finals})

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        s = "Print a DFTA\n"
        for (P, args), dst in self.rules.items():
            add = ""
            if len(args) > 0:
                add = "(" + ", ".join(map(str, args)) + ")"
            s += f"\t{P}{add} -> {dst}"
            if dst in self.finals:
                s += " (FINAL)"
            s += "\n"
        return s
