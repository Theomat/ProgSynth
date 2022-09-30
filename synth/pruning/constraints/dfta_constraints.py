from dataclasses import dataclass, field
from itertools import product
from typing import (
    Any,
    Callable,
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
from synth.pruning.constraints.parsing import (
    Token,
    TokenAllow,
    TokenAnything,
    TokenAtLeast,
    TokenAtMost,
    TokenFunction,
    TokenForceSubtree,
    TokenForbidSubtree,
    parse_specification,
)
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.grammars.det_grammar import DerivableProgram
from synth.syntax.grammars.cfg import CFG
from synth.syntax.type_system import Type


# ========================================================================================
# PARSING
# ========================================================================================


U = TypeVar("U")
V = TypeVar("V")

State = Tuple[DerivableProgram, Tuple[U, ...]]


@dataclass
class ProcessState:
    new_terminal_no: int = field(default=1)
    duplicate_from: Dict[State, State] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class Path(Generic[U]):
    predecessors: TList[Tuple[DerivableProgram, Tuple[U, ...], int]] = field(
        default_factory=lambda: []
    )

    def __hash__(self) -> int:
        return hash(tuple(self.predecessors))

    def __str__(self) -> str:
        if len(self) > 0:
            return "->".join(
                [
                    f"{P}(" + ",".join(map(str, args)) + ")"
                    for P, args, _ in self.predecessors
                ]
            )
        return "|"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.predecessors)

    def last(self) -> Tuple[DerivableProgram, Tuple[U, ...], int]:
        return self.predecessors[-1]

    def next(self, P: DerivableProgram, args: Tuple[U, ...], index: int) -> "Path[U]":
        return Path(self.predecessors + [(P, args, index)])

    def next_argument(self) -> None:
        P, args, i = self.predecessors[-1]
        self.predecessors[-1] = (P, args, i + 1)

    def map_fix(self, map: Callable[[U], U]) -> None:
        for k, (P, args, i) in enumerate(self.predecessors):
            self.predecessors[k] = (
                P,
                tuple(arg if j != i else map(arg) for j, arg in enumerate(args)),
                i,
            )


def __cfg2dfta__(
    grammar: CFG,
) -> DFTA[Tuple[Type, int], DerivableProgram]:
    StateT = Tuple[Type, int]
    dfta_rules: Dict[Tuple[DerivableProgram, Tuple[StateT, ...]], StateT] = {}
    max_depth = grammar.max_program_depth()
    all_cases: Dict[
        Tuple[int, Tuple[Type, ...]], Set[Tuple[Tuple[Type, int], ...]]
    ] = {}
    for S in grammar.rules:
        for P in grammar.rules[S]:
            args = grammar.rules[S][P][0]
            if len(args) == 0:
                dfta_rules[(P, ())] = (P.type, 0)
            else:
                key = (len(args), tuple([arg[0] for arg in args]))
                if key not in all_cases:
                    all_cases[key] = set(
                        [
                            tuple(x)
                            for x in product(
                                *[
                                    [(arg[0], j) for j in range(max_depth)]
                                    for arg in args
                                ]
                            )
                        ]
                    )
                for nargs in all_cases[key]:
                    dfta_rules[(P, nargs)] = (
                        P.type.returns(),
                        max(i for _, i in nargs) + 1,
                    )
    r = grammar.type_request.returns()
    return DFTA(dfta_rules, {(r, x) for x in range(max_depth)})


def __augment__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    relevant: TList[Tuple[Path[Tuple[Type, U]], Tuple[Type, U]]],
) -> Tuple[
    DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram],
    TList[Tuple[Path[Tuple[Type, Tuple[U, int]]], Tuple[Type, Tuple[U, int]]]],
]:
    new_dfta = DFTA(
        {
            (P, tuple((arg[0], (arg[1], 0)) for arg in args)): (dst[0], (dst[1], 0))
            for (P, args), dst in grammar.rules.items()
        },
        {(t, (q, 0)) for t, q in grammar.finals},
    )
    new_relevant = [
        (
            Path(
                [
                    (P, tuple([(arg[0], (arg[1], 0)) for arg in args]), i)
                    for P, args, i in path.predecessors
                ]
            ),
            (S[0], (S[1], 0)),
        )
        for path, S in relevant
    ]

    return new_dfta, new_relevant


def __count__(
    dfta: DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram],
    relevant: TList[
        Tuple[Path[Tuple[Type, Tuple[U, int]]], Tuple[Type, Tuple[U, int]]]
    ],
    count: int,
    to_count: TList[DerivableProgram],
    at_most: bool,
) -> Tuple[
    DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram],
    TList[Tuple[Path[Tuple[Type, Tuple[U, int]]], Tuple[Type, Tuple[U, int]]]],
]:
    maxi = count + (1 if at_most else 0)
    # We duplicate rules in order to count
    all_alternatives = lambda s: [(s[0], (s[1][0], i)) for i in range(maxi + 1)]
    for (P, args), dst in list(dfta.rules.items()):
        possibles = [all_alternatives(arg) for arg in args]
        del dfta.rules[(P, args)]
        for new_args in product(*possibles):
            total = sum(arg[1][1] for arg in new_args)
            if P in to_count:
                total += 1
            dfta.rules[(P, new_args)] = (dst[0], (dst[1][0], min(total, maxi)))

    # We update finals
    if at_most:
        for q in list(dfta.finals):
            for new_alt in all_alternatives(q):
                dfta.finals.add(new_alt)
    else:
        dfta.finals = {(q[0], (q[1][0], count)) for q in dfta.finals}
    # Now we need some kind of reset thing or to remove things
    are_equals = lambda a, b: a[0] == b[0] and a[1][0] == b[1][0]
    for path, path_end in relevant:
        if len(path) > 0:
            pP, pargs, i = path.last()
            for (P, args), dst in list(dfta.rules.items()):
                if pP == P and all(
                    are_equals(args[k], pargs[k]) for k in range(len(pargs))
                ):
                    if at_most:
                        # Remove rules that go over the count
                        if args[i][1][1] == maxi:
                            del dfta.rules[(P, args)]
                    else:
                        # other subtrees should not count so we do not count them
                        if any(k != i and arg[1][1] > 0 for k, arg in enumerate(args)):
                            dfta.rules[(P, args)] = (dst[0], (dst[1][0], args[i][1][1]))
        elif at_most:
            q = (path_end[0], (path_end[1][0], count))
            if q in dfta.finals:
                dfta.finals.remove(q)
        else:
            pass  # TODO: think
    # Now we need to duplicate the paths
    # TODO: check if that works
    next_relevant = []
    for path, path_end in relevant:
        if len(path) > 0:
            pred_possibles: TList[
                TList[
                    Tuple[DerivableProgram, Tuple[Tuple[Type, Tuple[U, int]], ...], int]
                ]
            ] = []
            for preP, pre_args, i in path.predecessors:
                candidates: TList[
                    Tuple[DerivableProgram, Tuple[Tuple[Type, Tuple[U, int]], ...], int]
                ] = []
                for (P, args) in dfta.rules:
                    if P == preP and all(
                        are_equals(args[k], pre_args[k]) for k in range(len(pre_args))
                    ):
                        candidates.append((P, args, i))
                pred_possibles.append(candidates)
            all_paths: TList[Path] = []
            for new_pred in product(*pred_possibles):
                all_paths.append(Path(list(new_pred)))
            for path in all_paths:
                P, args, i = path.last()
                next_relevant.append((path, args[i]))
        else:
            for (P, args), dst in dfta.rules.items():
                if are_equals(path_end, dst):
                    next_relevant.append((Path(), dst))

    return dfta, next_relevant


def __process__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    token: Token,
    sketch: bool,
    relevant: Optional[TList[Tuple[Path, Tuple[Type, U]]]] = None,
    level: int = 0,
    pstate: Optional[ProcessState] = None,
) -> Tuple[
    DFTA[Tuple[Type, Any], DerivableProgram], TList[Tuple[Path, Tuple[Type, U]]]
]:
    pstate = pstate or ProcessState()
    out_grammar: DFTA = grammar
    print("\t" * level, "processing:", token, "relevant:", relevant)
    if isinstance(token, TokenFunction):
        if relevant is None:
            # Compute relevant depending on sketch or not
            if sketch:
                relevant = [(Path(), q) for q in grammar.finals]
                grammar, relevant = __process__(
                    grammar, token.function, sketch, relevant, level, pstate
                )
            else:
                relevant = []
                for (P, args), dst in grammar.rules.items():
                    if P in token.function.allowed:
                        new_elem: Tuple[Path, Tuple[Type, U]] = (
                            Path(),
                            dst,
                        )
                        if new_elem not in relevant:
                            relevant.append(new_elem)
        else:
            # TODO: save or something alike
            # So here we have correct paths
            grammar, relevant = __process__(
                grammar, token.function, sketch, relevant, level, pstate
            )

        # Go from relevant to first argument context
        arg_relevant: TList[Tuple[Path, Tuple[Type, U]]] = []
        print("\t" * level, "relevant:", relevant)
        for path, end in relevant:
            for (P, args), dst in grammar.rules.items():
                if P in token.function.allowed and dst == end:
                    arg_relevant.append((path.next(P, args, 0), args[0]))
        print("\t" * level, "arg relevant:", arg_relevant)
        for arg in token.args:
            grammar, arg_relevant = __process__(
                grammar, arg, sketch, arg_relevant, level + 1, pstate
            )
            next_relevant = []
            for path, end in arg_relevant:
                if len(path) == 0:
                    next_relevant.append((path, end))
                    continue
                path.next_argument()
                P, args, i = path.last()
                if i >= len(args):
                    continue
                next_relevant.append((path, args[i]))
            arg_relevant = next_relevant
        out_grammar = grammar  # type trick
        return out_grammar, arg_relevant

    elif isinstance(token, TokenAllow):
        assert relevant is not None
        # We need to augment grammar to tell that this we detected this the correct thing
        out_grammar, out_relevant = __augment__(grammar, relevant)
        old2new = lambda s: (s[0], (s[1][0], 1))
        relevant = []
        old_finals: Set[Tuple[Type, Tuple[U, int]]] = {q for q in out_grammar.finals}
        producables: Set[Tuple[Type, Tuple[U, int]]] = {
            dst for (P, args), dst in out_grammar.rules.items() if P in token.allowed
        }
        # Create duplicates for the context
        for path, path_end in out_relevant:
            new_end = old2new(path_end)
            for (P, p_args), p_dst in list(out_grammar.rules.items()):
                if any(arg == path_end for arg in p_args) and (
                    sketch or len(path) == 0 or path.last()[0] != P
                ):
                    possibles = [
                        [arg] if arg != path_end else [arg, new_end] for arg in p_args
                    ]
                    for new_args in product(*possibles):
                        out_grammar.rules[(P, new_args)] = p_dst
        # Now add initial transition
        for path, path_end in out_relevant:
            new_end = old2new(path_end)
            for (P, p_args), p_dst in list(out_grammar.rules.items()):
                if p_dst == path_end and P in token.allowed:
                    out_grammar.rules[(P, p_args)] = new_end

            # Now we have to go back the track
            if len(path) == 0:
                if path_end in out_grammar.finals:
                    out_grammar.finals.remove(path_end)
                    out_grammar.finals.add(new_end)
                relevant.append((path, new_end))  # type: ignore
            else:
                P, p_args, i = path.last()
                if p_args[i] not in producables:
                    continue
                print("\t" * level, "path", path, "end", path_end)
                candidate_predecessors = []
                for P, p_args, i in path.predecessors:
                    old_dst = out_grammar.rules[(P, p_args)]
                    new_dst = old2new(old_dst) if sketch else old_dst
                    possibles = [
                        [arg, old2new(arg)] if j != i else [old2new(arg)]
                        for j, arg in enumerate(p_args)
                    ]
                    candidates = []
                    for new_args in product(*possibles):
                        out_grammar.rules[(P, new_args)] = new_dst
                        candidates.append((P, new_args, i))
                        print("\t", P, new_args, "->", new_dst)
                    candidate_predecessors.append(candidates)
                    if old_dst in old_finals:
                        out_grammar.finals.add(new_dst)
                        if old_dst in out_grammar.finals and sketch:
                            out_grammar.finals.remove(old_dst)

                P, p_args, i = path.last()
                if not sketch:
                    del out_grammar.rules[(P, p_args)]
                for new_preds in product(*candidate_predecessors):
                    new_path = Path(list(new_preds))
                    _, dst_args, i = new_path.last()
                    relevant.append((new_path, dst_args[i]))
            # relevant.append((path, new_end))  # type: ignore

        # We don't need to go deeper in relevant since this is an end node
        out_grammar.reduce()
        print(out_grammar)
        print("\t" * level, "out_relevant", relevant)
        return out_grammar, relevant

    elif isinstance(token, (TokenAtMost, TokenAtLeast)):
        assert relevant is not None
        out_grammar, out_relevant = __augment__(grammar, relevant)
        out_grammar, out_relevant = __count__(
            out_grammar,
            out_relevant,
            token.count,
            token.to_count,
            isinstance(token, TokenAtMost),
        )
        # What happens now?
        if isinstance(token, TokenAtMost):
            # We need to reset count at some points
            for path, path_end in out_relevant:
                pass
            pass
        # TODO
        print(out_grammar)
        out_grammar.reduce()
        print("\t" * level, "out_relevant:", out_relevant)
        return out_grammar, out_relevant  # type: ignore

    elif isinstance(token, TokenForbidSubtree):
        return __process__(
            grammar, TokenAtMost(token.forbidden, 0), sketch, relevant, level, pstate
        )
    elif isinstance(token, TokenForceSubtree):
        return __process__(
            grammar, TokenAtLeast(token.forced, 1), sketch, relevant, level, pstate
        )
    elif isinstance(token, TokenAnything):
        assert relevant is not None
        return grammar, relevant
    assert False, f"Not implemented: {token}"


def add_dfta_constraints(
    current_grammar: CFG,
    constraints: Iterable[str],
    sketch: bool = False,
    progress: bool = True,
) -> DFTA[Set, DerivableProgram]:
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
    dfta = __cfg2dfta__(current_grammar)
    dfta.reduce()

    if progress:
        pbar = tqdm.tqdm(total=len(parsed_constraints), desc="constraints", smoothing=1)
    pstate = ProcessState()
    for constraint in parsed_constraints:
        dfta = __process__(dfta, constraint, sketch, pstate=pstate)[0]
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
    # print(dfta)
    dfta.reduce()
    print(dfta)
    return dfta  # type: ignore # .minimise()  # type: ignore
