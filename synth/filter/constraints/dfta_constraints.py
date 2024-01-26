from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List as TList,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import tqdm
from synth.filter.constraints.parsing import (
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
    if max_depth == -1:
        for S in grammar.rules:
            for P in grammar.rules[S]:
                args = grammar.rules[S][P][0]
                if len(args) == 0:
                    dfta_rules[(P, ())] = (P.type, 0)
                else:
                    key = (len(args), tuple([arg[0] for arg in args]))
                    if key not in all_cases:
                        all_cases[key] = set([tuple([(arg[0], 0) for arg in args])])
                    for nargs in all_cases[key]:
                        dfta_rules[(P, nargs)] = (
                            S[0],
                            0,
                        )
    else:
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
                        new_depth = max(i for _, i in nargs) + 1
                        if new_depth >= max_depth:
                            continue
                        dfta_rules[(P, nargs)] = (
                            S[0],
                            new_depth,
                        )
    r = grammar.type_request.returns()
    dfta = DFTA(
        dfta_rules, {(r, x) for x in range(max_depth)} if max_depth > 0 else {(r, 0)}
    )
    dfta.reduce()
    return dfta


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


def __get_tuple_val__(t: Tuple[U, int], index: int) -> int:
    if index == -1:
        return t[1]
    return __get_tuple_val__(t[0], index + 1)  # type: ignore


def __tuple_len__(t: Tuple[U, ...]) -> int:
    if isinstance(t, tuple):
        return 1 + __tuple_len__(t[0])  # type: ignore
    return 1


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


def __filter__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    check: Callable[
        [
            DerivableProgram,
            Tuple[Tuple[Type, Tuple[U, int]], ...],
            Tuple[Type, Tuple[U, int]],
        ],
        bool,
    ],
) -> DFTA[Tuple[Type, U], DerivableProgram]:
    out_grammar: DFTA[Tuple[Type, U], DerivableProgram] = DFTA(
        {}, {q for q in grammar.finals}
    )
    # Whenever the pattern is correct we tag with 1
    for (P, p_args), p_dst in grammar.rules.items():
        if check(P, p_args, p_dst):  # type: ignore
            out_grammar.rules[(P, p_args)] = p_dst
    out_grammar.finals = out_grammar.finals.intersection(out_grammar.rules.values())
    return out_grammar


def __match__(
    args: Tuple[Tuple[Type, Tuple[U, int]], ...],
    dst: Tuple[Type, Tuple[U, int]],
    primitive_check: int,
    indices: TList[int],
    should_check: TList[bool],
) -> bool:
    if __get_tuple_val__(dst[1], -primitive_check - 2) != 1:
        return False
    for arg, state_index, check in zip(args, indices, should_check):
        if not check:
            continue
        if __get_tuple_val__(arg[1], -state_index - 2) != 1:
            return False

    return True


def __process__(
    grammar: DFTA[Tuple[Type, U], DerivableProgram],
    token: Token,
    local: bool,
    level: int = 0,
) -> DFTA[Tuple[Type, Any], DerivableProgram]:
    # print("\t" * level, "processing:", token)
    if isinstance(token, TokenFunction):
        has_check = []
        grammar = __process__(grammar, token.function, local, level + 1)
        lengths = [__tuple_len__(list(grammar.finals)[0][1])]  # type: ignore
        for arg in token.args:
            grammar = __process__(grammar, arg, local, level + 1)
            cur_len = __tuple_len__(list(grammar.finals)[0][1])  # type: ignore
            has_check.append(cur_len - lengths[-1] > 0)
            lengths.append(cur_len)

        indices = [lengths[-1] - l for l in lengths]
        primitive_check = indices.pop(0)
        out_grammar = __tag__(
            grammar,
            lambda _, args, dst: __match__(
                args, dst, primitive_check, indices, has_check
            ),
        )

    elif isinstance(token, TokenAllow):
        # We need to augment grammar to tell that this we detected this the correct thing
        allowed_P = token.allowed
        out_grammar = __tag__(grammar, lambda P, _, __: P in allowed_P)

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
        else:
            allowed = [token.count]
        out_grammar = __tag__(
            out_grammar, lambda _, __, state: state[1][0][-1] in allowed  # type: ignore
        )

    elif isinstance(token, TokenForbidSubtree):
        return __process__(grammar, TokenAtMost(token.forbidden, 0), local, level)
    elif isinstance(token, TokenForceSubtree):
        return __process__(grammar, TokenAtLeast(token.forced, 1), local, level)
    elif isinstance(token, TokenAnything):
        return grammar
    else:
        assert False, f"Not implemented token: {token}({type(token)}) [level={level}]"

    if level == 0:
        if not local:
            out_grammar.finals = {q for q in out_grammar.finals if q[1][-1] == 1}
        else:
            assert isinstance(
                token, TokenFunction
            ), f"Unsupported topmost token for local constraint"
            out_grammar = __filter__(
                out_grammar,
                lambda P, _, dst: P not in token.function.allowed or dst[1][-1] == 1,
            )
            # out_grammar.finals = out_grammar.finals.intersection(set(out_grammar.rules.keys()))
    return out_grammar


def add_dfta_constraints(
    current_grammar: Union[CFG, DFTA[Tuple[Type, Any], DerivableProgram]],
    constraints: Iterable[str],
    sketch: Optional[str] = None,
    progress: bool = True,
) -> DFTA[Tuple[Type, Set], DerivableProgram]:
    """
    Add constraints to the specified grammar.

    If sketch is True the constraints are for sketches otherwise they are pattern like.
    If progress is set to True use a tqdm progress bar.

    """
    constraint_plus = [(int("var" in c), c) for c in constraints]
    constraint_plus.sort(reverse=True)
    parsed_constraints = [
        parse_specification(constraint, current_grammar)  # type: ignore
        for _, constraint in constraint_plus
    ]
    dfta = None
    pbar = None
    if progress:
        pbar = tqdm.tqdm(
            total=len(parsed_constraints) + int(sketch is not None),
            desc="constraints",
            smoothing=1,
        )
    base = (
        __cfg2dfta__(current_grammar)
        if isinstance(current_grammar, CFG)
        else current_grammar
    )
    for constraint in parsed_constraints:
        # Skip empty allow since it means the primitive was not recognized
        if isinstance(constraint, TokenAnything) or (
            isinstance(constraint, TokenFunction)
            and len(constraint.function.allowed) == 0
        ):
            if pbar:
                pbar.update(1)
            continue
        a = __process__(base, constraint, True)
        if dfta is None:
            dfta = a
        else:
            a.reduce()
            dfta = dfta.read_product(a.minimise())
        dfta.reduce()
        dfta = dfta.minimise()  # type: ignore
        if pbar:
            pbar.update(1)
    if sketch is not None:
        a = __process__(
            base,
            parse_specification(sketch, current_grammar),  # type: ignore
            False,
        )
        if dfta is None:
            dfta = a
        else:
            a.reduce()
            dfta = dfta.read_product(a.minimise())  # type: ignore
        if pbar:
            pbar.update(1)
        dfta.reduce()
        dfta = dfta.minimise()  # type: ignore
    if pbar:
        pbar.close()
    return dfta or base
