from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List as TList,
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


U = TypeVar("U")
V = TypeVar("V")


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
) -> DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram]:
    new_dfta = DFTA(
        {
            (P, tuple((arg[0], (arg[1], 0)) for arg in args)): (dst[0], (dst[1], 0))
            for (P, args), dst in grammar.rules.items()
        },
        {(t, (q, 0)) for t, q in grammar.finals},
    )
    return new_dfta


def __tuples_get__(t: Tuple[U, int], index: int) -> int:
    if index == 0:
        return t[1]
    return __tuples_get__(t[0], index - 1)  # type: ignore


def __match__(
    args: TList[Tuple[Type, Tuple[U, int]]], to_check: TList[Tuple[int, TList[int]]]
) -> bool:
    for arg, (i, allowed) in zip(args, to_check):
        if len(allowed) > 0 and __tuples_get__(arg[1], i) not in allowed:
            return False
    return True


def __count__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    count: int,
    to_count: TList[DerivableProgram],
    at_most: bool,
) -> DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram]:
    dfta = __augment__(grammar)
    maxi = count + (1 if at_most else 0)
    # We duplicate rules in order to count
    all_alternatives = lambda s: [(s[0], (s[1][0], i)) for i in range(maxi + 1)]
    for (P, args), dst in list(dfta.rules.items()):
        possibles = [all_alternatives(arg) for arg in args]
        for new_args in product(*possibles):
            total = sum(arg[1][1] for arg in new_args)
            if P in to_count:
                total += 1
            dfta.rules[(P, new_args)] = (dst[0], (dst[1][0], min(total, maxi)))
    # We duplicate finals as well
    for q in list(dfta.finals):
        for new_q in all_alternatives(q):
            dfta.finals.add(new_q)
    return dfta


def __tag__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    check: Callable[
        [
            DerivableProgram,
            Tuple[Tuple[Type, Tuple[U, int]], ...],
            Tuple[Type, Tuple[U, int]],
        ],
        bool,
    ],
) -> DFTA[Tuple[Type, Tuple[U, int]], DerivableProgram]:
    out_grammar = __augment__(grammar)
    tag_state = lambda s: (s[0], (s[1][0], 1))
    added = set()
    # Whenever the pattern is correct we tag with 1
    for (P, p_args), p_dst in list(out_grammar.rules.items()):
        if check(P, p_args, p_dst):
            out_grammar.rules[(P, p_args)] = tag_state(p_dst)
            added.add(p_dst)
    # We also need to be able to consume the new tagged states like the others
    for (P, p_args), p_dst in list(out_grammar.rules.items()):
        if any(arg in added for arg in p_args):
            possibles = [
                [arg] if arg not in added else [arg, tag_state(arg)] for arg in p_args
            ]
            for new_args in product(*possibles):
                out_grammar.rules[(P, new_args)] = p_dst
    for q in added.intersection(out_grammar.finals):
        out_grammar.finals.add(tag_state(q))
    return out_grammar


def __process__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    token: Token,
    sketch: bool,
    level: int = 0,
) -> Tuple[DFTA[Tuple[Type, Any], DerivableProgram], int, TList[int]]:
    print("\t" * level, "processing:", token)
    if isinstance(token, TokenFunction):
        possibles = []
        added_length = []
        for arg in token.args:
            grammar, no_added, corrects = __process__(grammar, arg, sketch, level + 1)
            added_length.append(no_added)
            possibles.append(corrects)
        # Compile what we need to check
        to_check = []
        length_so_far = 1
        for allowed, added in zip(possibles[::-1], added_length[::-1]):
            to_check.append((length_so_far, allowed))
            length_so_far += added
        to_check = to_check[::-1]
        # Augment grammar and mark 1 when right pattern
        allowed_P = token.function.allowed
        out_grammar = __tag__(
            grammar,
            lambda P, p_args, __: P in allowed_P and __match__(list(p_args), to_check),
        )

        # We need to do something if we are the final node
        if level == 0:
            if sketch:
                out_grammar.finals = {q for q in out_grammar.finals if q[1][1] == 1}
            else:
                # TODO: think
                for (P, p_args), p_dst in list(out_grammar.rules.items()):
                    if P in token.function.allowed and p_dst[1][1] == 0:
                        pass

        return out_grammar, length_so_far, [1]

    elif isinstance(token, TokenAllow):
        # We need to augment grammar to tell that this we detected this the correct thing
        allowed_P = token.allowed
        out_grammar = __tag__(grammar, lambda P, _, __: P in allowed_P)
        return out_grammar, 1, [1]

    elif isinstance(token, (TokenAtMost, TokenAtLeast)):
        out_grammar = __count__(
            grammar,
            token.count,
            token.to_count,
            isinstance(token, TokenAtMost),
        )
        allowed = []
        if isinstance(token, TokenAtMost):
            allowed = list(range(token.count + 1))
            print("\tallowed:", allowed)
        else:
            allowed = [token.count]
        return out_grammar, 1, allowed

    elif isinstance(token, TokenForbidSubtree):
        return __process__(grammar, TokenAtMost(token.forbidden, 0), sketch, level)
    elif isinstance(token, TokenForceSubtree):
        return __process__(grammar, TokenAtLeast(token.forced, 1), sketch, level)
    elif isinstance(token, TokenAnything):
        return grammar, 0, []
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
    for constraint in parsed_constraints:
        dfta = __process__(dfta, constraint, sketch)[0]
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()

    print("=" * 80)
    print(dfta)
    print("-" * 80)
    # print(dfta)
    dfta.reduce()
    print(dfta)
    return dfta  # type: ignore # .minimise()  # type: ignore
    # return dfta.minimise()  # type: ignore
