from typing import Dict, List, Optional, Tuple
import bisect
import numpy as np

from synth.syntax.concrete.concrete_pcfg import ConcretePCFG, Context
from synth.syntax.program import Program

NodeData = Tuple[float, List[Context], List[Program], List[Context]]
"""(probability, next_contexts, program, deriv_history)"""


def __common_prefix__(a: List[Context], b: List[Context]) -> List[Context]:
    if a == b:
        return a
    candidates = []
    if len(a) > 1:
        candidates.append(__common_prefix__(a[1:], b))
        if a[0] == b[0]:
            candidates.append([a[0]] + __common_prefix__(a[1:], b[1:]))
    if len(b) > 1:
        candidates.append(__common_prefix__(a, b[1:]))
    # Take longest common prefix
    lentghs = [len(x) for x in candidates]
    if max(lentghs) == lentghs[0]:
        return candidates[0]
    return candidates[1]


def __pcfg_from__(
    original_pcfg: ConcretePCFG, group: List[NodeData]
) -> Tuple[List[Program], ConcretePCFG]:
    # print("=" * 60)
    # Find the common prefix to all
    min_prefix = group[0][-1]
    for _, _, _, deriv_prefix in group[1:]:
        min_prefix = __common_prefix__(min_prefix, deriv_prefix)
        # print("\t", x)

    # for _, _, x, _ in group:
    #     for _, _ , y, _ in group:
    #         print("x=", x, "y=",y, "compatible=", __are_compatible__(original_pcfg, x, y))
    assert min_prefix is not None
    # Extract the start symbol
    start = min_prefix.pop(0)
    # print("Min_prefix=", (start, min_prefix))
    # print("Max program prefix:", group[0][2])
    # print("Start=", start)
    # Mark all paths that should be filled
    to_fill: List[Context] = []
    rules: Dict[Context, Dict[Program, Tuple[List[Context], float]]] = {start: {}}
    for _, args, program, prefix in group:
        # Create rules to follow the path
        ctx_path: List = prefix[len(min_prefix) + 1 :]
        program_path: List = program[len(min_prefix) + 1 :]
        for i, P in enumerate(program_path):
            current = start if i == 0 else ctx_path[i - 1]
            if current not in rules:
                rules[current] = {}
            rules[current][P] = (
                original_pcfg.rules[current][P][0],
                10.0,
            )  # 10 because in logprobs and probs it doesn't make sense so easy to see an error
            # print("\t", current, "->", P)
        # If there is no further derivation
        if not args:
            continue
        # Next derivation should be filled
        # print("From", program, "args:", args)
        for arg in args:
            to_fill.append(arg)

    # At this point rules can generate all partial programs
    # Get the S to normalize by descending depth order
    to_normalise = sorted(list(rules.keys()), key=lambda x: x[-1])
    # print("To normalise:", to_normalise)
    # print("To fill:", [x[1][0] for x in to_fill])

    # Build rules from to_fill
    while to_fill:
        S = to_fill.pop()
        # print("\tFilling:", S)
        rules[S] = {}
        for P in original_pcfg.rules[S]:
            args, w = original_pcfg.rules[S][P]
            rules[S][P] = (args[:], w)  # copy list
            for arg in args:
                if arg not in rules:
                    to_fill.append(arg)
    # So far so good works as expected

    # At this point we have all the needed rules
    # However, the probabilites are incorrect
    while to_normalise:
        S = to_normalise.pop()
        # Compute the updated probabilities
        for P in list(rules[S]):
            args, _ = rules[S][P]
            # We have the following equation:
            # (1) w = old_w * remaining_fraction
            old_w = original_pcfg.rules[S][P][1]
            remaining_fraction: float = 1
            # If there is a next derivation use it to compute the remaining_fraction
            if args:
                N = args[-1]
                remaining_fraction = sum(rules[N][K][1] for K in rules[N])
            # Update according to Equation (1)
            rules[S][P] = args, old_w * remaining_fraction

        # The updated probabilities may not sum to 1 so we need to normalise them
        # But let ConcretePCFG do it with clean=True

    min_depth: int = start[-1]
    program_prefix = group[0][2][: len(min_prefix)]
    # print("Program prefix=", program_prefix)

    # Now our min_prefix may be something like MAP, which takes 2 arguments
    # but the generators need to know that the start symbol is simply not (?, (MAP, 1), 1)
    # that is there must be a (?, (MAP, 0), 1) generated afterwards

    # Compute missing arguments to generate
    derivations = min_prefix[:]
    program_path = program_prefix[::-1]
    stack: List[Tuple[Context, Program, int]] = []
    for S, P in zip(derivations, program_path):
        argsP, w = original_pcfg.rules[S][P]
        if stack:
            Sp, Pp, n = stack.pop()
            if n > 1:
                stack.append((Sp, Pp, n - 1))
        if len(argsP) > 0:
            stack.append((S, P, len(argsP)))
    # We are missing start so we have to do it manually
    if stack:
        Sp, Pp, n = stack.pop()
        if n > 1:
            stack.append((Sp, Pp, n - 1))

    # print("derivations=", derivations)
    # print("program_path=", program_path)
    # print("stack=", stack)
    # Stack contains all the HOLES
    l = len(min_prefix)
    i = -1
    while stack:
        S, P, n = stack.pop()
        start = S
        while True:
            St, Pt = derivations[i], program_path[i]
            i -= 1
            rules[St] = {Pt: (original_pcfg.rules[St][Pt][0], 1.0)}
            if St == S and Pt == P:
                break
        # rules[start] = {P: (original_pcfg.rules[start][P], 1.0)}
    l += i + 1

    # print("start=", start)
    min_depth = start[-1]
    program_prefix = group[0][2][:l]

    # Ensure rules are depth ordered
    rules = {key: rules[key] for key in sorted(list(rules.keys()), key=lambda x: x[-1])}

    # Update max depth
    max_depth: int = original_pcfg.max_program_depth - min_depth
    # print("Symbols=", list(rules.keys()))

    return program_prefix, ConcretePCFG(start, rules, max_depth, clean=True)


def __node_split__(pcfg: ConcretePCFG, node: NodeData) -> Tuple[bool, List[NodeData]]:
    """
    Split the specified node accordingly.

    Return: success, nodes:
    - True, list of children nodes
    - False, [node]
    """
    output: List[NodeData] = []
    prob, next_contexts, program, deriv_history = node
    # If there is no next then it means this node can't be split
    if len(next_contexts) == 0:
        return False, [node]
    new_context: Context = next_contexts.pop()
    for P in pcfg.rules[new_context]:
        args, p_prob = pcfg.rules[new_context][P]
        new_root = (
            prob * p_prob,
            next_contexts + args,
            program + [P],
            deriv_history + [new_context],
        )
        output.append(new_root)
    return True, output


def __split_nodes_until_quantity_reached__(
    pcfg: ConcretePCFG, quantity: int
) -> List[NodeData]:
    """
    Start from the root node and split most probable node until the threshold number of nodes is reached.
    """
    nodes: List[NodeData] = [(1.0, [pcfg.start], [], [])]
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


def __holes_of__(pcfg: ConcretePCFG, node: NodeData) -> List[Context]:
    stack = [pcfg.start]
    current = node[2][:]
    while current:
        S = stack.pop()
        P = current.pop()
        args = pcfg.rules[S][P][0]
        for arg in args:
            stack.append(arg)
    return stack


def __is_fixing_any_hole__(
    pcfg: ConcretePCFG, node: NodeData, holes: List[Context]
) -> bool:
    current = node[2][:]
    stack = [pcfg.start]
    while current:
        S = stack.pop()
        if S in holes:
            return True
        P = current.pop()
        args = pcfg.rules[S][P][0]
        for arg in args:
            stack.append(arg)
    return False


def __are_compatible__(pcfg: ConcretePCFG, node1: NodeData, node2: NodeData) -> bool:
    """
    Two nodes prefix are compatible if one does not fix a context for the other.
    e.g. a -> b -> map -> *  and c -> b -> map -> +1 -> * are incompatible.

    In both cases map have the same context (bigram context) which is ((predecessor=b, argument=0), depth=2) thus are indistinguishables.
    However in the former all derivations are allowed in this context whereas in the latter +1 must be derived.
    Thus we cannot create a CFG that enables both.
    """
    holes1 = __holes_of__(pcfg, node1)
    if __is_fixing_any_hole__(pcfg, node2, holes1):
        return False
    holes2 = __holes_of__(pcfg, node2)
    return not __is_fixing_any_hole__(pcfg, node1, holes2)


def __all_compatible__(
    pcfg: ConcretePCFG, node: NodeData, group: List[NodeData]
) -> bool:
    return all(__are_compatible__(pcfg, node, node2) for node2 in group)


def __try_split_node_in_group__(
    pcfg: ConcretePCFG, prob_groups: List[List], group_index: int
) -> bool:
    group_a: List[NodeData] = prob_groups[group_index][1]
    # Sort group by ascending probability
    group_a_bis = sorted(group_a, key=lambda x: x[0])
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
    pcfg: ConcretePCFG, prob_groups: List[List], group_index: int
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
            pa: float = node_a[0]
            reduced_prob: float = prob - pa
            # Try all swaps
            for k, node_b in enumerate(group_b):
                pb: float = node_b[0]
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
            pb = node_b[0]
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
        prob_groups[group_index][1] -= node_a[0]
        prob_groups[j][0].append(node_a)
        prob_groups[j][1] += node_a[0]

    node_b = prob_groups[j][0].pop(l)
    prob_groups[j][1] -= node_b[0]
    prob_groups[group_index][0].append(node_b)
    prob_groups[group_index][1] += node_b[0]

    __percolate_down__(prob_groups, -1)
    __percolate_up__(prob_groups, group_index)


def __split_into_nodes__(
    pcfg: ConcretePCFG, splits: int, desired_ratio: float = 2
) -> Tuple[List[List[NodeData]], float]:
    nodes = __split_nodes_until_quantity_reached__(pcfg, splits)

    # Create groups
    groups: List[List[NodeData]] = []
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

    # Improve
    improved = True
    masses: List[float] = [np.sum([x[0] for x in group]) for group in groups]
    prob_groups = sorted([[g, p] for g, p in zip(groups, masses)], key=lambda x: x[1])  # type: ignore
    ratio: float = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    while improved and ratio > desired_ratio:
        improved = False
        for i in range(splits - 1):
            swap = __find_swap_for_group__(pcfg, prob_groups, i)
            if swap:
                improved = True
                __apply_swap__(prob_groups, i, swap)
                break
        if not improved:
            for i in range(splits - 1, 0, -1):
                improved = __try_split_node_in_group__(pcfg, prob_groups, i)
                if improved:
                    break
        ratio = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    return [g for g, _ in prob_groups], ratio  # type: ignore


def split(
    pcfg: ConcretePCFG, splits: int, desired_ratio: float = 1.1
) -> Tuple[List[Tuple[List[Program], ConcretePCFG]], float]:
    """
    Currently use exchange split.
    Parameters:
    desired_ratio: the max ratio authorized between the most probable group and the least probable pcfg

    Return:
    a list of Tuple[prefix program, ConcretePCFG]
    the reached threshold
    """
    if splits == 1:
        return [([], pcfg)], 1
    assert desired_ratio > 1, "The desired ratio must be > 1!"
    groups, ratio = __split_into_nodes__(pcfg, splits, desired_ratio)
    return [__pcfg_from__(pcfg, group) for group in groups], ratio
