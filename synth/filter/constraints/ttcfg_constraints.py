from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List as TList,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import tqdm
from synth.filter.constraints.parsing import (
    Token,
    TokenAllow,
    TokenAtLeast,
    TokenAtMost,
    TokenFunction,
    TokenForceSubtree,
    TokenForbidSubtree,
    parse_specification,
)
from synth.syntax.automata.dfa import DFA
from synth.syntax.grammars.det_grammar import DerivableProgram
from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.type_system import Type


# ========================================================================================
# PARSING
# ========================================================================================


U = TypeVar("U")
V = TypeVar("V")

State = Tuple[Type, Tuple[Tuple[U, int], V]]
Info = TList[Tuple[Type, Tuple[U, int]]]
DerElment = Tuple[Type, Tuple[U, int]]


@dataclass
class ProcessState:
    new_terminal_no: int = field(default=1)
    duplicate_from: Dict[State, State] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class Path(Generic[U, V]):
    predecessors: TList[
        Tuple[Tuple[Type, Tuple[Tuple[U, int], V]], DerivableProgram]
    ] = field(default_factory=lambda: [])

    def __hash__(self) -> int:
        return hash(tuple(self.predecessors))

    def __str__(self) -> str:
        if len(self) > 0:
            end = f"->{self.predecessors[-1][1]}"
            return "->".join([f"{S[1]}" for S, P in self.predecessors]) + end
        return "-"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.predecessors)

    def last(self) -> Tuple[Tuple[Type, Tuple[Tuple[U, int], V]], DerivableProgram]:
        return self.predecessors[-1]

    def fix_last(self, S: Tuple[Type, Tuple[Tuple[U, int], V]]) -> None:
        self.predecessors[-1] = (S, self.predecessors[-1][1])

    def next(
        self, S: Tuple[Type, Tuple[Tuple[U, int], V]], P: DerivableProgram
    ) -> "Path[U, V]":
        return Path(self.predecessors + [(S, P)])


Save = Tuple[
    Tuple[Type, Tuple[Tuple[U, int], V]], TList[Tuple[DerivableProgram, int, V]], Info
]


def __make_save__(
    grammar: TTCFG[Tuple[U, int], V], path: Path[U, V], S: State, info: Info
) -> Save:
    hist: TList[Tuple[DerivableProgram, int, V]] = []
    tuples = path.predecessors[:]
    nexts = [ctx for ctx, _ in path.predecessors[1:]] + [S]
    save = (path.predecessors[0][0], hist, info)
    while tuples:
        S, P = tuples.pop(0)
        nextS = nexts.pop(0)
        der = (nextS[0], nextS[1][0])
        derlst, _ = grammar.rules[S][P]
        found = False
        for i, SS in enumerate(derlst):
            if SS == der:
                found = True
                hist.append((P, i, nextS[1][1]))
                break
        assert found
    return save


def __restore_save__(
    grammar: TTCFG[Tuple[U, int], V], save: Save
) -> Tuple[Path[U, V], State, Info]:
    start, hist, info = save
    path: Path[U, V] = Path()
    while hist:
        P, i, v = hist.pop(0)
        derlst, _ = grammar.rules[start][P]
        der = derlst[i]
        next_S = (der[0], (der[1], v))
        path = path.next(start, P)
        start = next_S
    return path, start, info


def __dfa_start_from_any__(
    grammar: TTCFG[Tuple[U, int], V],
    relevant: TList[Tuple[Path, Tuple[Type, Tuple[Tuple[U, int], V]], Info]],
    state: ProcessState,
) -> Tuple[
    DFA[int, Tuple[Type, Tuple[Tuple[U, int], V]]], TList[Tuple[Path, State, Info]]
]:
    # Create DFA that self loops
    rules: Dict[int, Dict[Tuple[Type, Tuple[Tuple[U, int], V]], int]] = {
        0: {S: 0 for S in grammar.rules},
        1: {},
    }
    # Now for only the relevant states we will need to not self loops
    relevant_cpy = relevant[:]
    # Redirection is to fix the path towards S if it has been changed during iteration
    redirections: Dict[State, State] = {}
    # Already done avoids conflicts when there are multiple identical derivations where only the from state changes
    already_done: Dict[Tuple[State, Tuple[Type, Tuple[U, int]]], State] = {}
    relevant = []
    while relevant_cpy:
        path, S, info = relevant_cpy.pop()
        if len(path) > 0:
            parent_S, parent_P = path.last()
            if parent_S in redirections:
                parent_S = redirections[parent_S]
                path.fix_last(parent_S)
            key = (parent_S, (S[0], S[1][0]))
        else:
            key = None
        if key in already_done:
            tmpS = already_done[key]
            new_S = (tmpS[0], (tmpS[1][0], S[1][1]))
            grammar.rules[new_S] = {
                P: (grammar.rules[S][P][0][:], grammar.rules[S][P][1])
                for P in grammar.rules[S]
            }
        elif S in redirections:
            new_S = redirections[S]
        else:
            new_S = __duplicate__(grammar, S, state)
            if key is not None:
                already_done[key] = new_S
            redirections[S] = new_S
        if len(path) > 0:
            __redirect__(grammar, parent_S, parent_P, S, new_S)
        else:
            grammar.start = new_S
        rules[0][new_S] = 1
        outcomes = grammar.possible_outcomes_after(new_S)
        if len(path) > 0:
            parent_S, parent_P = path.last()
            derlst, _ = grammar.rules[parent_S][parent_P]
            der = (new_S[0], new_S[1][0])

            added, found = False, False
            nextS = der
            for SS in derlst:
                if SS == der:
                    found = True
                elif found:
                    nextS = SS
                    added = True
                    break
            if not added:
                nextS = info[0]
            for v in outcomes:
                rules[1][(nextS[0], (nextS[1], v))] = 0

        relevant.append((path, new_S, info))
    return DFA(0, rules), relevant


def __count_dfa__(
    grammar: TTCFG[Tuple[U, int], V],
    dfa: DFA[int, Tuple[Type, Tuple[Tuple[U, int], V]]],
    to_count: TList[DerivableProgram],
    count: int,
) -> DFA[int, Tuple[Tuple[Type, Tuple[Tuple[U, int], V]], DerivableProgram]]:
    all_primitives: TList[DerivableProgram] = list(grammar.primitives_used())
    all_primitives += grammar.variables()

    rules: Dict[int, Dict[DerivableProgram, int]] = {}
    start_count = count
    while count > 0:
        rules[count] = {
            P: count - 1 if P in to_count else count for P in all_primitives
        }
        count -= 1
    # count == 0
    rules[count] = {P: count for P in all_primitives if P not in to_count}
    counter = DFA(start_count, rules).map_states(lambda i: i + len(dfa.states) - 1)
    new_rules: Dict[
        int, Dict[Tuple[Tuple[Type, Tuple[Tuple[U, int], V]], DerivableProgram], int]
    ] = {}

    for u in dfa.rules:
        new_rules[u] = {}
        for S in dfa.rules[u]:
            for P in grammar.rules[S]:
                new_rules[u][(S, P)] = 0 if dfa.rules[u][S] == 0 else counter.start

    for u in counter.rules:
        new_rules[u] = {}
        for S in grammar.rules:
            for P in counter.rules[u]:
                new_rules[u][(S, P)] = (
                    0
                    if S in dfa.rules[1] and dfa.rules[1][S] == 0
                    else counter.rules[u][P]
                )
    return DFA(0, new_rules)


def __preprocess_grammar__(grammar: TTCFG[U, V]) -> TTCFG[Tuple[U, int], V]:
    """
    Goes from U to Tuple[U, int] by just making tuples of (U, 0)
    """
    new_rules: Dict[State, Dict[DerivableProgram, Tuple[TList[DerElment], V]]] = {}
    for S in grammar.rules:
        u, v = S[1]
        SS = (S[0], ((u, 0), v))
        new_rules[SS] = {}
        for P in grammar.rules[S]:
            derlst, state = grammar.rules[S][P]
            new_rules[SS][P] = ([(t, (d, 0)) for t, d in derlst], state)
    return TTCFG(
        (grammar.start[0], ((grammar.start[1][0], 0), grammar.start[1][1])),
        new_rules,
        clean=False,
    )


def __redirect__(
    grammar: TTCFG[Tuple[U, int], V],
    parent_S: State,
    parent_P: DerivableProgram,
    old: State,
    new: State,
) -> None:
    """
    Redirect parent_S -> parent_P : ...old...
    to parent_S -> parent_P : ...new...
    """
    # Redirect original derivation
    derlst, state = grammar.rules[parent_S][parent_P]
    old_element = (old[0], old[1][0])
    new_element = (new[0], new[1][0])
    new_derlst = [el if el != old_element else new_element for el in derlst]
    grammar.rules[parent_S][parent_P] = new_derlst, state


def __duplicate__(
    grammar: TTCFG[Tuple[U, int], V], S: State, state: ProcessState
) -> State:
    """
    Duplicate S and return the copy new_S
    """
    up, v = S[1]
    new_S = (S[0], ((up[0], state.new_terminal_no), v))
    state.new_terminal_no += 1
    if S in state.duplicate_from:
        # print("found origin using:", state.duplicate_from[S], "instead of", S)
        # print(
        #     "\t",
        #     len(grammar.rules[state.duplicate_from[S]]),
        #     " instead of",
        #     len(grammar.rules[S]),
        # )
        S = state.duplicate_from[S]
    state.duplicate_from[new_S] = S
    grammar.rules[new_S] = {
        P: (grammar.rules[S][P][0][:], grammar.rules[S][P][1]) for P in grammar.rules[S]
    }
    return new_S


def __forbid_subtree__(
    grammar: TTCFG[Tuple[U, int], V],
    parent_S: State,
    parent_P: DerivableProgram,
    S: State,
    forbidden: TList[DerivableProgram],
    state: ProcessState,
    done: Optional[Set[State]] = None,
    info: Optional[TList[Tuple[Type, Tuple[U, int]]]] = None,
) -> None:
    new_S = __duplicate__(grammar, S, state)
    __redirect__(grammar, parent_S, parent_P, S, new_S)
    done = done or set()
    for P in sorted(grammar.rules[S].keys(), key=lambda P: str(P)):
        if P in forbidden:
            # Delete forbidden
            del grammar.rules[new_S][P]
        else:
            # Recursive calls
            info, nextS = grammar.derive(info or grammar.start_information(), S, P)
            # Nothing to do
            if nextS not in grammar.rules:
                continue
            __forbid_subtree__(grammar, S, P, nextS, forbidden, state, done, info)


def __process__(
    grammar: TTCFG[Tuple[U, int], V],
    token: Token,
    sketch: bool,
    relevant: Optional[TList[Tuple[Path, State, Info]]] = None,
    level: int = 0,
    state: Optional[ProcessState] = None,
) -> Tuple[
    TTCFG[Tuple[U, int], Any],
    Dict[Path[U, V], TList[Set[V]]],
    TList[Tuple[Path, State, Info]],
]:
    assert not isinstance(
        token, (TokenAtLeast, TokenForceSubtree)
    ), "Unsupported constraint for TTCFG(safe, det)"
    out_grammar: TTCFG[Tuple[U, int], Any] = grammar
    state = state or ProcessState()
    possible_new_states: Dict[Path[U, V], TList[Set[V]]] = defaultdict(list)
    # print(
    #     "\t" * level,
    #     "processing:",
    #     token,
    #     "len(relevant)=",
    #     len(relevant) if relevant else 0,
    # )
    # if relevant is not None:
    #     for path, S, info in relevant:
    #         print("\t" * level, " path:", path, "S:", S)
    if isinstance(token, TokenFunction):
        if relevant is None:
            # Compute relevant depending on sketch or not
            if sketch:
                relevant = [(Path(), grammar.start, grammar.start_information())]
                grammar, _, __ = __process__(
                    grammar, token.function, sketch, relevant, level, state
                )
                relevant = []
                relevant.append((Path(), grammar.start, grammar.start_information()))
            else:
                relevant = []
                for S in grammar.rules:
                    for P in grammar.rules[S]:
                        if P in token.function.allowed:
                            relevant.append((Path(), S, grammar.start_information()))
                            break
        else:
            saves = [
                __make_save__(grammar, path, S, info) for path, S, info in relevant
            ]
            # So here we have correct paths
            grammar, _, __ = __process__(
                grammar, token.function, sketch, relevant, level, state
            )
            # However we have restricted the possible functions so we renamed our paths
            # We need to fix that
            new_relevant = [__restore_save__(grammar, save) for save in saves]

            relevant = new_relevant
        # Go from relevant to first argument context
        arg_relevant: TList[Tuple[Path, State, Info]] = []
        # print("\t" * level, "building arg relevant")
        for path, S, info in relevant:
            # print("\t" * level, "  path:", path)
            for P in grammar.rules[S]:
                if P in token.function.allowed:
                    new_path = path.next(S, P)
                    new_info, new_S = grammar.derive(info, S, P)
                    arg_relevant.append((new_path, new_S, new_info))

        for argno, arg in enumerate(token.args):
            grammar, possible_states, new_relevant = __process__(
                grammar, arg, sketch, arg_relevant, level + 1, state
            )
            if isinstance(arg, TokenAtMost):
                arg_relevant = new_relevant
            next_relevant: TList[Tuple[Path, State, Info]] = []
            for path, S, info in arg_relevant:
                pS, pP = path.last()
                derlst = grammar.rules[pS][pP][0]
                if argno + 1 >= len(derlst):
                    continue
                t, u = derlst[argno + 1]
                for states_list in possible_states[path]:
                    for v in states_list:
                        new_S = (t, (u, v))
                        next_relevant.append((path, new_S, info))
            arg_relevant = next_relevant
        out_grammar = grammar
    elif isinstance(token, TokenAllow):
        assert relevant is not None
        relevant_cpy = relevant[:]
        # Redirection is to fix the path towards S if it has been changed during iteration
        redirections: Dict[State, State] = {}
        # Already done avoids conflicts when there are multiple identical derivations where only the from state changes
        already_done: Dict[Tuple[State, Tuple[Type, Tuple[U, int]]], State] = {}
        relevant = []
        while relevant_cpy:
            path, S, info = relevant_cpy.pop()
            if len(path) > 0:
                parent_S, parent_P = path.last()
                if parent_S in redirections:
                    parent_S = redirections[parent_S]
                    if parent_P not in grammar.rules[parent_S]:
                        # The path that we took was actually deleted by the constraints
                        continue
                    path.fix_last(parent_S)
                # print(
                #     "\t" * (level + 1), "parent:", parent_S , "->", parent_P, "=>", S)
                key = (parent_S, (S[0], S[1][0]))
            else:
                key = None
            should_del = True
            if key in already_done:
                # print("\t" * (level + 1), "copy")
                tmpS = already_done[key]
                new_S = (tmpS[0], (tmpS[1][0], S[1][1]))
                grammar.rules[new_S] = {
                    P: (grammar.rules[S][P][0][:], grammar.rules[S][P][1])
                    for P in grammar.rules[S]
                }
            elif S in redirections:
                # print("\t" * (level + 1), "redirection")
                new_S = redirections[S]
                should_del = False
            else:
                # print("\t" * (level + 1), "duplicate")
                new_S = __duplicate__(grammar, S, state)
                if key is not None:
                    already_done[key] = new_S
                redirections[S] = new_S
            if should_del:
                # Delete forbidden
                for P in list(grammar.rules[new_S].keys()):
                    if P not in token.allowed:
                        # print("\t" * (level + 3), "del:", new_S, "->", P)
                        del grammar.rules[new_S][P]
            if len(path) > 0:
                __redirect__(grammar, parent_S, parent_P, S, new_S)
            else:
                grammar.start = new_S

            # print("\t" * (level + 1), "from:", S, "to", new_S)
            relevant.append((path, new_S, info))
    elif isinstance(token, TokenAtMost):
        assert relevant is not None
        # Create a DFA that recognises the path
        detector, relevant = __dfa_start_from_any__(grammar, relevant, state)
        # Create a DFA that counts primitives
        final_dfa = __count_dfa__(grammar, detector, token.to_count, token.count)
        out_grammar = grammar * final_dfa
        # Relevant contains old paths since we augmented the type of the grammar
        # We need to create the new relevant
        new_relevant = []
        # easier to go from saves we only need to map the starting one
        ssaves = [
            (path, __make_save__(grammar, path, S, info)) for path, S, info in relevant
        ]
        for or_path, save in ssaves:
            start, hist, info = save
            assert len(hist) == len(or_path)
            # Let's hope info is useless and we can just not care
            found = 0
            for SS in out_grammar.rules:
                old_S = (SS[0], (SS[1][0], SS[1][1][0]))
                if old_S == start:
                    try:
                        path, SSS, Sinfo = __restore_save__(
                            out_grammar, (SS, hist[:], info)  # type: ignore
                        )
                        assert len(path) == len(
                            or_path
                        ), f"or_path={or_path} path={path}"
                        new_relevant.append((path, SSS, Sinfo))
                        found += 1
                    except KeyError:
                        pass
        relevant = new_relevant
    elif isinstance(token, TokenForbidSubtree):
        assert relevant is not None
        for path, S, info in relevant:
            parent_S, parent_P = path.last()
            __forbid_subtree__(grammar, parent_S, parent_P, S, token.forbidden, state)
    # Compute valid possible new states
    assert relevant is not None
    for path, S, info in relevant:
        all_new_states = grammar.possible_outcomes_after(S)
        possible_new_states[path].append(all_new_states)
    return out_grammar, possible_new_states, relevant


def add_constraints(
    current_grammar: TTCFG[U, V],
    constraints: Iterable[str],
    sketch: Optional[str] = None,
    progress: bool = True,
) -> TTCFG[Tuple[U, int], Any]:
    """
    Add constraints to the specified grammar.

    If sketch is True the constraints are for sketches otherwise they are pattern like.
    If progress is set to True use a tqdm progress bar.

    """
    constraint_plus = [(int("var" in c), c) for c in constraints]
    constraint_plus.sort(reverse=True)
    parsed_constraints = [
        parse_specification(constraint, current_grammar)
        for _, constraint in constraint_plus
    ]
    preprocessed = __preprocess_grammar__(current_grammar)

    if progress:
        pbar = tqdm.tqdm(
            total=len(parsed_constraints) + int(sketch is not None),
            desc="constraints",
            smoothing=1,
        )
    state = ProcessState()
    for constraint in parsed_constraints:
        preprocessed = __process__(preprocessed, constraint, False, state=state)[0]
        if progress:
            pbar.update(1)
    if sketch is not None:
        preprocessed = __process__(
            preprocessed,
            parse_specification(sketch, current_grammar),
            True,
            state=state,
        )[0]
        if progress:
            pbar.update(1)
    preprocessed.clean()
    if progress:
        pbar.close()
    return preprocessed
