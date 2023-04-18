from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import bisect
from dataclasses import dataclass, field
import copy

import numpy as np

from synth.syntax.grammars.tagged_det_grammar import DerivableProgram
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.grammars.u_cfg import UCFG
from synth.syntax.program import Constant, Primitive, Program, Variable
from synth.syntax.type_system import Type, UnknownType

U = TypeVar("U")


@dataclass(order=True, frozen=True)
class _Node(Generic[U]):
    probability: float
    for_next_derivation: Tuple[List[Tuple[Type, U]], Tuple[Type, U]] = field(
        compare=False
    )
    program: List[Program] = field(compare=False)
    derivation_history: List[Tuple[Type, U]] = field(compare=False)
    choices: List[List[Tuple[Type, U]]] = field(compare=False)


def __node_split__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], node: _Node[U]
) -> Tuple[bool, List[_Node[U]]]:
    """
    Split the specified node accordingly.

    Return: success, nodes:
    - True, list of children nodes
    - False, [node]
    """
    output: List[_Node[U]] = []
    info, S = node.for_next_derivation
    # If there is no next then it means this node can't be split
    if S not in pcfg.tags:
        return False, [node]
    for P in pcfg.rules[S]:
        p_prob = pcfg.probabilities[S][P]
        # format is (info_state_stack, current_info_state, possibles)
        next_derivations = pcfg.derive(info, S, P)
        # Skip failed derivations
        if len(next_derivations) == 0:
            continue
        for possible in next_derivations:
            new_root = _Node(
                node.probability * p_prob[tuple(possible[-1])],  # type: ignore
                (possible[0], possible[1]),
                node.program + [P],
                node.derivation_history + [S],
                node.choices + [possible[-1]],
            )
            output.append(new_root)
    return True, output


def __split_nodes_until_quantity_reached__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], quantity: int
) -> List[_Node[U]]:
    """
    Start from the root node and split most probable node until the threshold number of nodes is reached.
    """
    nodes: List[_Node[U]] = []
    for key, prob in pcfg.start_tags.items():
        nodes.append(_Node(prob, (pcfg.start_information(), key), [], [], []))
    while len(nodes) < quantity:
        i = 1
        success, new_nodes = __node_split__(pcfg, nodes.pop())
        while not success:
            i += 1
            nodes.append(new_nodes[0])
            success, new_nodes = __node_split__(pcfg, nodes.pop(-i))
        for new_node in new_nodes:
            insertion_index: int = bisect.bisect(nodes, new_node)
            nodes.insert(insertion_index, new_node)

    return nodes


def __all_compatible__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    node: _Node[U],
    group: List[_Node[U]],
) -> bool:
    return True  # all(__are_compatible__(pcfg, node, node2) for node2 in group)


def __try_split_node_in_group__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    prob_groups: List[List],
    group_index: int,
) -> bool:
    group_a: List[_Node[U]] = prob_groups[group_index][1]
    # Sort group by ascending probability
    group_a_bis = sorted(group_a, key=lambda x: x.probability)
    # Try splitting a node until success
    i = 1
    success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
    while not success and i < len(group_a):
        i += 1
        success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
    if i >= len(group_a):
        return False
    # Success, remove old node
    group_a.pop(-i)
    # Add new nodes
    for new_node in new_nodes:
        group_a.append(new_node)
    return True


def __find_swap_for_group__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    prob_groups: List[List],
    group_index: int,
) -> Optional[Tuple[int, Optional[int], int]]:
    max_prob: float = prob_groups[-1][1]
    min_prob: float = prob_groups[0][1]
    group_a, prob = prob_groups[group_index]
    best_swap: Optional[Tuple[int, Optional[int], int]] = None
    current_score: float = max_prob / prob

    candidates = (
        list(range(len(prob_groups) - 1, group_index, -1))
        if group_index == 0
        else [len(prob_groups) - 1]
    )

    for i in candidates:
        group_b, prob_b = prob_groups[i]
        for j, node_a in enumerate(group_a):
            pa: float = node_a.probability
            reduced_prob: float = prob - pa
            # Try all swaps
            for k, node_b in enumerate(group_b):
                pb: float = node_b.probability
                if (
                    pb < pa
                    or not __all_compatible__(pcfg, node_a, group_b)
                    or not __all_compatible__(pcfg, node_b, group_a)
                ):
                    continue
                new_mass_b: float = prob_b - pb + pa
                mini = min_prob if group_index > 0 else reduced_prob + pb
                maxi = (
                    max(new_mass_b, prob_groups[-2][1])
                    if j == len(prob_groups) - 1
                    else max_prob
                )
                new_score = maxi / mini
                if new_score < current_score:
                    best_swap = (i, j, k)
                    current_score = new_score
        # Consider taking something from b
        for k, node_b in enumerate(group_b):
            if not __all_compatible__(pcfg, node_b, group_a):
                continue
            pb = node_b.probability
            if prob + pb > max_prob:
                new_score = (prob + pb) / min_prob
            else:
                new_score = max_prob / (prob + pb)
            if new_score < current_score:
                best_swap = (i, None, k)
                current_score = new_score
    return best_swap


def __percolate_down__(prob_groups: List[List], group_index: int) -> None:
    index = group_index
    p = prob_groups[group_index][1]
    while index > 0 and prob_groups[index - 1][1] > p:
        prob_groups[index - 1], prob_groups[index] = (
            prob_groups[index],
            prob_groups[index - 1],
        )
        index -= 1


def __percolate_up__(prob_groups: List[List], group_index: int) -> None:
    index = group_index
    p = prob_groups[group_index][1]
    while index < len(prob_groups) - 2 and prob_groups[index + 1][1] < p:
        prob_groups[index + 1], prob_groups[index] = (
            prob_groups[index],
            prob_groups[index + 1],
        )
        index += 1


def __apply_swap__(
    prob_groups: List[List], group_index: int, swap: Tuple[int, Optional[int], int]
) -> None:
    j, k, l = swap
    # App
    if k:
        node_a = prob_groups[group_index][0].pop(k)
        prob_groups[group_index][1] -= node_a.probability
        prob_groups[j][0].append(node_a)
        prob_groups[j][1] += node_a.probability

    node_b = prob_groups[j][0].pop(l)
    prob_groups[j][1] -= node_b.probability
    prob_groups[group_index][0].append(node_b)
    prob_groups[group_index][1] += node_b.probability

    __percolate_down__(prob_groups, -1)
    __percolate_up__(prob_groups, group_index)


def __split_into_nodes__(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    splits: int,
    threshold: float,
) -> Tuple[List[List[_Node[U]]], float]:
    nodes = __split_nodes_until_quantity_reached__(pcfg, splits)
    # Create groups
    groups: List[List[_Node[U]]] = []
    for node in nodes[:splits]:
        groups.append([node])
    for node in nodes[splits:]:
        # Add to first compatible group
        added = False
        for group in groups:
            if __all_compatible__(pcfg, node, group):
                group.append(node)
                added = True
                break
        assert added
    masses: List[float] = [np.sum([x.probability for x in group]) for group in groups]
    prob_groups = sorted([[g, p] for g, p in zip(groups, masses)], key=lambda x: x[1])  # type: ignore
    ratio: float = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    made_progress = True
    while ratio > threshold and made_progress:
        made_progress = False
        for i in range(splits - 1):
            swap = __find_swap_for_group__(pcfg, prob_groups, i)
            if swap:
                made_progress = True
                __apply_swap__(prob_groups, i, swap)
                break
        if not made_progress:
            for i in range(splits - 1, 0, -1):
                made_progress = __try_split_node_in_group__(pcfg, prob_groups, i)
                if made_progress:
                    break
        ratio = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    return [g for g, _ in prob_groups], ratio  # type: ignore


def __common_prefix__(
    a: List[Tuple[Type, U]], b: List[Tuple[Type, U]]
) -> List[Tuple[Type, U]]:
    if a == b:
        return a
    candidates = []
    if len(a) > 1:
        candidates.append(__common_prefix__(a[1:], b))
        if len(b) >= 1 and a[0] == b[0]:
            candidates.append([a[0]] + __common_prefix__(a[1:], b[1:]))
    if len(b) > 1:
        candidates.append(__common_prefix__(a, b[1:]))
    # Take longest common prefix
    lentghs = [len(x) for x in candidates]
    if len(lentghs) == 0:
        return []
    if max(lentghs) == lentghs[0]:
        return candidates[0]
    return candidates[1]


def __create_path__(
    rules: Dict[
        Tuple[Type, Tuple[U, int]],
        Dict[DerivableProgram, List[List[Tuple[Type, Tuple[U, int]]]]],
    ],
    probabilities: Dict[
        Tuple[Type, Tuple[U, int]],
        Dict[DerivableProgram, Dict[List[Tuple[Type, Tuple[U, int]]], float]],
    ],
    original_pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    Slist: List[Tuple[Type, U]],
    Plist: List[Program],
    Vlist: List[List[Tuple[Type, U]]],
    map_state: Callable[[Tuple[Type, U]], Tuple[Type, Tuple[U, int]]],
    original_start: Tuple[Type, U],
    to_normalise: List[
        Tuple[List[Tuple[Type, U]], Tuple[Type, U], Program, List[Tuple[Type, U]]]
    ],
) -> List[Tuple[Type, U]]:
    # print("\tCREATING A PATH:", Plist)
    info = original_pcfg.start_information()
    for i, (S, P, v) in enumerate(zip(Slist, Plist, Vlist)):
        if i == 0:
            S = original_start
        # print(f"\t\tpath:{S} -> {P} : {v}")
        to_normalise.append((info, S, P, v))
        if i > 0:
            info.pop(0)
        derivation = original_pcfg.derive_specific(info, S, P, v)  # type: ignore
        assert derivation
        next_derivation, current = derivation
        # Update derivations
        assert isinstance(P, (Primitive, Variable, Constant))
        # print(f"\t\tpath current:{current} next:{next_derivation}")
        Sp = map_state(S)
        mapped_v = [map_state(x) for x in v]
        if Sp not in rules:
            rules[Sp] = {P: []}
            probabilities[Sp] = {P: {}}
        if P not in rules[Sp]:
            rules[Sp][P] = []
            probabilities[Sp][P] = {}
        rules[Sp][P].append(mapped_v)
        probabilities[Sp][P][tuple(mapped_v)] = original_pcfg.probabilities[S][P][tuple(v)]  # type: ignore
        info = [current] + next_derivation
    return info


def __pcfg_from__(
    original_pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    group: List[_Node[U]],
) -> ProbUGrammar[
    Tuple[U, int], List[Tuple[Type, Tuple[U, int]]], List[Tuple[Type, Tuple[U, int]]]
]:

    # print()
    # print("=" * 60)
    # print("NODE")
    # for node in group:
    #     print("\t", node.program)
    # Find the common prefix to all
    min_prefix = copy.deepcopy(group[0].derivation_history)
    for node in group[1:]:
        min_prefix = __common_prefix__(min_prefix, node.derivation_history)
    # print("MIN PREFIX:", min_prefix)

    # Function to map states automatically
    rule_nos = [0]
    mapping: Dict[Tuple[Type, U], Tuple[Type, Tuple[U, int]]] = {}

    def map_state(s: Tuple[Type, U]) -> Tuple[Type, Tuple[U, int]]:
        o = mapping.get(s, None)
        if o is not None:
            # print("\t", s, "=>", o)
            return o
        # print(s, "does not exist in mapping:", set(mapping.keys()))
        mapping[s] = (s[0], (s[1], rule_nos[0]))
        rule_nos[0] += 1
        return mapping[s]

    # New start states
    starts = {map_state(s) for s in original_pcfg.grammar.starts}

    # Extract the start symbol
    start = min_prefix.pop()

    rules: Dict[
        Tuple[Type, Tuple[U, int]],
        Dict[DerivableProgram, List[List[Tuple[Type, Tuple[U, int]]]]],
    ] = {}
    probabilities: Dict[
        Tuple[Type, Tuple[U, int]],
        Dict[DerivableProgram, Dict[List[Tuple[Type, Tuple[U, int]]], float]],
    ] = {}
    start_probs: Dict[Tuple[Type, Tuple[U, int]], float] = {}
    # List of non-terminals + info we need to normalise weirdly
    to_normalise: List[
        Tuple[List[Tuple[Type, U]], Tuple[Type, U], Program, List[Tuple[Type, U]]]
    ] = []
    # Our min_prefix may be something like (int, 1, (+, 1))
    # which means we already chose +
    # But it is not in the PCFG
    # Thus we need to add it
    # In the general case we may as well have + -> + -> + as prefix this whole prefix needs to be added
    original_start = start

    # We also need to mark all contexts that should be filled
    to_fill: List[Tuple[List[Tuple[Type, U]], Tuple[Type, U]]] = []
    if len(min_prefix) > 0:
        Slist = group[0].derivation_history[: len(min_prefix) + 1]
        Plist = group[0].program[: len(min_prefix) + 1]
        Vlist = group[0].choices[: len(min_prefix) + 1]
        rem = __create_path__(
            rules,
            probabilities,
            original_pcfg,
            Slist,
            Plist,
            Vlist,
            map_state,
            Slist[0],
            to_normalise,
        )
        if rem and not isinstance(rem[0][0], UnknownType):
            to_fill.append((rem, rem.pop(0)))
        original_start = Slist[-1]

    # Now we need to make a path from the common prefix to each node's prefix

    for node in group:
        program, prefix = (
            node.program,
            node.derivation_history,
        )
        # Create rules to follow the path
        i = prefix.index(original_start)
        if len(min_prefix) > 0:
            i += 1
        ctx_path = prefix[i:]
        program_path = program[i:]
        v_path = node.choices[i:]
        # print(prefix, "START:", original_start)
        if len(ctx_path) > 0:
            ctx_path[0] = original_start
            rem = __create_path__(
                rules,
                probabilities,
                original_pcfg,
                ctx_path,
                program_path,
                v_path,
                map_state,
                original_start,
                to_normalise,
            )
            if rem and not isinstance(rem[0][0], UnknownType):
                to_fill.append((rem, rem.pop(0)))

    # print("BEFORE FILLING")
    # print("to_fill:", to_fill)
    # print(UCFG(starts, rules, clean=False))

    computed: Dict[
        Tuple[Type, Tuple[U, int]],
        Dict[DerivableProgram, Dict[Tuple[Tuple[Type, Tuple[U, int]], ...], float]],
    ] = defaultdict(lambda: defaultdict(dict))
    # Build rules from to_fill
    while to_fill:
        info, S = to_fill.pop()
        Sp = map_state(S)
        already_done = Sp in rules
        if not already_done:
            rules[Sp] = {}
            probabilities[Sp] = {}
        # print("\t", S,"//", Sp, ":")
        for P in original_pcfg.rules[S]:
            possibles = original_pcfg.rules[S][P]
            new_list = []
            if P not in probabilities[Sp]:
                probabilities[Sp][P] = {}
            # print("\t\t", P, "=>")
            for v in possibles:
                if not already_done:
                    prob = original_pcfg.probabilities[S][P][tuple(v)]  # type: ignore
                    mapped_v = [map_state(x) for x in v]
                    probabilities[Sp][P][tuple(mapped_v)] = prob  # type: ignore
                    # print(f"\t\t\t{prob}: {v} // {mapped_v}")
                    computed[Sp][P][tuple(mapped_v)] = prob
                    new_list.append(mapped_v)
                a = original_pcfg.derive_specific(info, S, P, v)
                if (
                    a
                    and (already_done or map_state(a[1]) not in rules)
                    and not isinstance(a[1][0], UnknownType)
                ):
                    to_fill.append(a)
            if not already_done:
                rules[Sp][P] = new_list

    # Now we can already have the new grammar
    new_grammar = UCFG(starts, rules, clean=True)
    # print("FINAL GRAMMAR BEFORE CLEANING")
    # print(new_grammar)
    # new_grammar.clean()
    # At this point we have all the needed rules
    # However, the probabilites are incorrect
    while to_normalise:
        for el in list(to_normalise):
            info, S, cP, v = el
            assert isinstance(cP, (Primitive, Variable, Constant))
            Sp = map_state(S)
            if Sp not in probabilities:
                probabilities[Sp] = {}
            if cP not in probabilities[Sp]:
                probabilities[Sp][cP] = {}
            # print("\t", Sp, "->", P)
            derivation = original_pcfg.derive_specific(info, S, cP, v)
            assert derivation
            _, current = derivation
            # Compute the updated probabilities
            new_prob = 0.0
            old_w = original_pcfg.probabilities[S][cP][tuple(v)]  # type: ignore
            if isinstance(current[0], UnknownType):
                new_prob = 1
            else:
                missed = False
                currentP = map_state(current)
                count = 0
                for Pp, v_dict in computed[currentP].items():
                    if Pp not in computed[currentP]:
                        missed = True
                        break
                    for cv, p in v_dict.items():
                        if cv not in v_dict:
                            missed = True
                            break
                        new_prob += p
                        count += 1
                if missed or count == 0:
                    continue
                # print("for", S, "=>", P, "@", v)
                # print("\tprob:", new_prob, "count:", count, "missed:", missed)
            # Update according to Equation (1)
            tmapped_v = tuple(map_state(x) for x in v)
            # print("\t\t", Sp, "->", P)
            probabilities[Sp][cP][tmapped_v] = old_w * new_prob  # type: ignore
            computed[Sp][cP][tmapped_v] = old_w * new_prob
            to_normalise.remove(el)

    for start in original_pcfg.start_tags:
        Sp = map_state(start)
        if Sp in new_grammar.starts:
            start_probs[Sp] = original_pcfg.start_tags[start]
        # The updated probabilities may not sum to 1 so we need to normalise them
        # But let ProbDetGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]] do it with clean=True
    # print("START", probabilities)
    probabilities = {
        S: {
            P: {v: p for v, p in dicoV.items() if list(v) in new_grammar.rules[S][P]}
            for P, dicoV in dicoP.items()
            if P in new_grammar.rules[S]
        }
        for S, dicoP in probabilities.items()
        if S in new_grammar.rules
    }
    # for S, dicoP in probabilities.items():
    #     print("\t", S, ":")
    #     for P, possibles in dicoP.items():
    #         print("\t\t", P, " =>", possibles)
    #     print()
    grammar = ProbUGrammar(new_grammar, probabilities, start_probs)
    grammar.normalise()
    # print(grammar)
    # print()
    # print("=" * 80)
    # print()
    return grammar


# @overload
# def split(
#     pcfg: ProbDetGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], splits: int, desired_ratio: float = 1.1
# ) -> Tuple[List[ProbDetGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]]], float]:
#     pass

# @overload
# def split(
#     pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], splits: int, desired_ratio: float = 1.1
# ) -> Tuple[List[ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]]], float]:
#     pass

# def split(
#     pcfg: Union[ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]], ProbDetGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]]], splits: int, desired_ratio: float = 1.1
# ) -> Union[Tuple[List[ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]]], float], Tuple[List[ProbDetGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]]], float]]:
def split(
    pcfg: ProbUGrammar[U, List[Tuple[Type, U]], List[Tuple[Type, U]]],
    splits: int,
    desired_ratio: float = 1.1,
) -> Tuple[
    List[
        ProbUGrammar[
            Tuple[U, int],
            List[Tuple[Type, Tuple[U, int]]],
            List[Tuple[Type, Tuple[U, int]]],
        ]
    ],
    float,
]:
    """
    Currently use exchange split.
    Parameters:
    splits: the number of splits (must be > 1, otherwise the pcfg returned does not match the type signature since the input one is returned)
    desired_ratio: the max ratio authorized between the most probable group and the least probable pcfg

    Return:
    a list of probabilistic grammars
    the reached threshold
    """
    if splits == 1:
        return [pcfg], 1  # type: ignore
    assert desired_ratio > 1, "The desired ratio must be > 1!"
    groups, ratio = __split_into_nodes__(pcfg, splits, desired_ratio)
    return [__pcfg_from__(pcfg, group) for group in groups if len(group) > 0], ratio
