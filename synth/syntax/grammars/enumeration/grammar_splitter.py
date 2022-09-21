# from typing import Dict, Generic, List, Optional, Tuple, TypeVar
# import bisect
# from dataclasses import dataclass, field
# import copy

# import numpy as np

# from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar, DerivableProgram
# from synth.syntax.program import Program
# from synth.syntax.type_system import Type

# U = TypeVar("U")
# V = TypeVar("V")
# W = TypeVar("W")


# @dataclass(order=True, frozen=True)
# class Node(Generic[U, W]):
#     probability: float
#     for_next_derivation: Tuple[W, Tuple[Type, U]] = field(compare=False)
#     program: List[Program] = field(compare=False)
#     derivation_history: List[Tuple[Type, U]] = field(compare=False)


# def __common_prefix__(
#     a: List[Tuple[Type, U]], b: List[Tuple[Type, U]]
# ) -> List[Tuple[Type, U]]:
#     if a == b:
#         return a
#     candidates = []
#     if len(a) > 1:
#         candidates.append(__common_prefix__(a[1:], b))
#         if len(b) >= 1 and a[0] == b[0]:
#             candidates.append([a[0]] + __common_prefix__(a[1:], b[1:]))
#     if len(b) > 1:
#         candidates.append(__common_prefix__(a, b[1:]))
#     # Take longest common prefix
#     lentghs = [len(x) for x in candidates]
#     if len(lentghs) == 0:
#         return []
#     if max(lentghs) == lentghs[0]:
#         return candidates[0]
#     return candidates[1]


# def __adapt_ctx__(S: Tuple[Type, U], i: int) -> Tuple[Type, U]:
#     pred = S.predecessors[0]
#     return Tuple[Type, U](S.type, [(pred[0], i)] + S.predecessors[1:], S.depth)


# def __create_path__(
#     rules: PRules,
#     original_pcfg: ProbDetGrammar[U, V, W],
#     rule_no: int,
#     Slist: List[Tuple[Type, U]],
#     Plist: List[Program],
#     mapping: Dict[Tuple[Type, U], Tuple[Type, U]],
#     original_start: Tuple[Type, U],
# ) -> int:
#     for i, (S, P) in enumerate(zip(Slist, Plist)):
#         if i == 0:
#             S = original_start
#         derivations = original_pcfg.rules[S][P][0]
#         # Update derivations
#         new_derivations = []
#         for nS in derivations:
#             if nS not in Slist:
#                 new_derivations.append(nS)
#             else:
#                 if nS in mapping:
#                     new_derivations.append(mapping[nS])
#                 else:
#                     mS = __adapt_ctx__(nS, rule_no)
#                     mapping[nS] = mS
#                     new_derivations.append(mS)
#                     rule_no += 1
#         derivations = new_derivations
#         # Update current S
#         if i > 0:
#             S = mapping[S]
#         else:
#             S = Slist[0]
#         # Add rule
#         rules[S] = {}
#         rules[S][P] = derivations, 1
#     return rule_no


# def __pcfg_from__(
#     original_pcfg: ProbDetGrammar[U, V, W], group: List[Node]
# ) -> ProbDetGrammar[U, V, W]:
#     # Find the common prefix to all
#     min_prefix = copy.deepcopy(group[0].derivation_history)
#     for node in group[1:]:
#         min_prefix = __common_prefix__(min_prefix, node.derivation_history)

#     # Extract the start symbol
#     start = min_prefix.pop()

#     rules: Dict[Tuple[Type, U], Dict[Program, Tuple[List[Tuple[Type, U]], float]]] = {}
#     rule_no: int = (
#         max(
#             max(x[1] for x in key.predecessors) if key.predecessors else 0
#             for key in original_pcfg.rules
#         )
#         + 1
#     )
#     mapping: Dict[Tuple[Type, U], Tuple[Type, U]] = {}
#     # Our min_prefix may be something like (int, 1, (+, 1))
#     # which means we already chose +
#     # But it is not in the PCFG
#     # Thus we need to add it
#     # In the general case we may as well have + -> + -> + as prefix this whole prefix needs to be added
#     original_start = start
#     if len(min_prefix) > 0:
#         Slist = group[0].derivation_history[: len(min_prefix) + 1]
#         Plist = group[0].program[: len(min_prefix) + 1]
#         rule_no = __create_path__(
#             rules, original_pcfg, rule_no, Slist, Plist, mapping, Slist[0]
#         )
#         original_start = Slist[-1]
#         start = mapping[original_start]

#     # Now we need to make a path from the common prefix to each node's prefix
#     # We also need to mark all contexts that should be filled
#     to_fill: List[Tuple[Type, U]] = []
#     for node in group:
#         args, program, prefix = (
#             node.next_contexts,
#             node.program,
#             node.derivation_history,
#         )
#         # Create rules to follow the path
#         i = prefix.index(original_start)
#         ctx_path = prefix[i:]
#         program_path = program[i:]
#         if len(ctx_path) > 0:
#             ctx_path[0] = start
#             rule_no = __create_path__(
#                 rules,
#                 original_pcfg,
#                 rule_no,
#                 ctx_path,
#                 program_path,
#                 mapping,
#                 original_start,
#             )
#         # If there is no further derivation
#         if not args:
#             continue
#         # Next derivation should be filled
#         for arg in args:
#             to_fill.append(arg)

#     # At this point rules can generate all partial programs
#     # Get the S to normalize by descending depth order
#     to_normalise = sorted(list(rules.keys()), key=lambda x: -x.depth)

#     # Build rules from to_fill
#     while to_fill:
#         S = to_fill.pop()
#         rules[S] = {}
#         for P in original_pcfg.rules[S]:
#             args, w = original_pcfg.rules[S][P]
#             rules[S][P] = (args[:], w)  # copy list
#             for arg in args:
#                 if arg not in rules:
#                     to_fill.append(arg)
#     # At this point we have all the needed rules
#     # However, the probabilites are incorrect
#     while to_normalise:
#         S = to_normalise.pop()
#         if S not in original_pcfg.rules:
#             continue
#         # Compute the updated probabilities
#         for P in list(rules[S]):
#             args, _ = rules[S][P]
#             # We have the following equation:
#             # (1) w = old_w * remaining_fraction
#             old_w = original_pcfg.rules[S][P][1]
#             remaining_fraction: float = 1
#             # If there is a next derivation use it to compute the remaining_fraction
#             if args:
#                 N = args[-1]
#                 remaining_fraction = sum(rules[N][K][1] for K in rules[N])
#             # Update according to Equation (1)
#             rules[S][P] = args, old_w * remaining_fraction

#         # The updated probabilities may not sum to 1 so we need to normalise them
#         # But let ProbDetGrammar[U, V, W] do it with clean=True

#     start = original_pcfg.start

#     # Ensure rules are depth ordered
#     rules = {
#         key: rules[key] for key in sorted(list(rules.keys()), key=lambda x: x.depth)
#     }

#     return ProbDetGrammar[U, V, W](
#         start, rules, original_pcfg.max_program_depth, clean=True
#     )


# def __node_split__(
#     pcfg: ProbDetGrammar[U, V, W], node: Node
# ) -> Tuple[bool, List[Node]]:
#     """
#     Split the specified node accordingly.

#     Return: success, nodes:
#     - True, list of children nodes
#     - False, [node]
#     """
#     output: List[Node] = []
#     info, S = node.for_next_derivation
#     # If there is no next then it means this node can't be split
#     if S not in pcfg.tags:
#         return False, [node]
#     for P in pcfg.rules[S]:
#         p_prob = pcfg.probabilities[S][P]
#         next = pcfg.derive(info, S, P)
#         new_root = Node(
#             node.probability * p_prob,
#             next,
#             node.program + [P],
#             node.derivation_history + [S],
#         )
#         output.append(new_root)
#     return True, output


# def __split_nodes_until_quantity_reached__(
#     pcfg: ProbDetGrammar[U, V, W], quantity: int
# ) -> List[Node]:
#     """
#     Start from the root node and split most probable node until the threshold number of nodes is reached.
#     """
#     nodes: List[Node] = [Node(1.0, (pcfg.start_information(), pcfg.start), [], [])]
#     while len(nodes) < quantity:
#         i = 1
#         success, new_nodes = __node_split__(pcfg, nodes.pop())
#         while not success:
#             i += 1
#             nodes.append(new_nodes[0])
#             success, new_nodes = __node_split__(pcfg, nodes.pop(-i))
#         for new_node in new_nodes:
#             insertion_index: int = bisect.bisect(nodes, new_node)
#             nodes.insert(insertion_index, new_node)

#     return nodes


# def __holes_of__(pcfg: ProbDetGrammar[U, V, W], node: Node) -> List[Tuple[Type, U]]:
#     stack = [pcfg.start]
#     current = node.program[:]
#     while current:
#         S = stack.pop()
#         P = current.pop(0)
#         args = pcfg.rules[S][P][0]
#         for arg in args:
#             stack.append(arg)
#     return stack


# def __is_fixing_any_hole__(
#     pcfg: ProbDetGrammar[U, V, W], node: Node, holes: List[Tuple[Type, U]]
# ) -> bool:
#     current = node.program[:]
#     stack = [pcfg.start]
#     while current:
#         S = stack.pop()
#         if S in holes:
#             return True
#         P = current.pop(0)
#         args = pcfg.rules[S][P][0]
#         for arg in args:
#             stack.append(arg)
#     return False


# def __are_compatible__(pcfg: ProbDetGrammar[U, V, W], node1: Node, node2: Node) -> bool:
#     """
#     Two nodes prefix are compatible if one does not fix a context for the other.
#     e.g. a -> b -> map -> *  and c -> b -> map -> +1 -> * are incompatible.

#     In both cases map have the same context (bigram context) which is ((predecessor=b, argument=0), depth=2) thus are indistinguishables.
#     However in the former all derivations are allowed in this context whereas in the latter +1 must be derived.
#     Thus we cannot create a CFG that enables both.
#     """
#     holes1 = __holes_of__(pcfg, node1)
#     if __is_fixing_any_hole__(pcfg, node2, holes1):
#         return False
#     holes2 = __holes_of__(pcfg, node2)
#     return not __is_fixing_any_hole__(pcfg, node1, holes2)


# def __all_compatible__(
#     pcfg: ProbDetGrammar[U, V, W], node: Node, group: List[Node]
# ) -> bool:
#     return all(__are_compatible__(pcfg, node, node2) for node2 in group)


# def __try_split_node_in_group__(
#     pcfg: ProbDetGrammar[U, V, W], prob_groups: List[List], group_index: int
# ) -> bool:
#     group_a: List[Node] = prob_groups[group_index][1]
#     # Sort group by ascending probability
#     group_a_bis = sorted(group_a, key=lambda x: x.probability)
#     # Try splitting a node until success
#     i = 1
#     success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
#     while not success and i < len(group_a):
#         i += 1
#         success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
#     if i >= len(group_a):
#         return False
#     # Success, remove old node
#     group_a.pop(-i)
#     # Add new nodes
#     for new_node in new_nodes:
#         group_a.append(new_node)
#     return True


# def __find_swap_for_group__(
#     pcfg: ProbDetGrammar[U, V, W], prob_groups: List[List], group_index: int
# ) -> Optional[Tuple[int, Optional[int], int]]:
#     max_prob: float = prob_groups[-1][1]
#     min_prob: float = prob_groups[0][1]
#     group_a, prob = prob_groups[group_index]
#     best_swap: Optional[Tuple[int, Optional[int], int]] = None
#     current_score: float = max_prob / prob

#     candidates = (
#         list(range(len(prob_groups) - 1, group_index, -1))
#         if group_index == 0
#         else [len(prob_groups) - 1]
#     )

#     for i in candidates:
#         group_b, prob_b = prob_groups[i]
#         for j, node_a in enumerate(group_a):
#             pa: float = node_a.probability
#             reduced_prob: float = prob - pa
#             # Try all swaps
#             for k, node_b in enumerate(group_b):
#                 pb: float = node_b.probability
#                 if (
#                     pb < pa
#                     or not __all_compatible__(pcfg, node_a, group_b)
#                     or not __all_compatible__(pcfg, node_b, group_a)
#                 ):
#                     continue
#                 new_mass_b: float = prob_b - pb + pa
#                 mini = min_prob if group_index > 0 else reduced_prob + pb
#                 maxi = (
#                     max(new_mass_b, prob_groups[-2][1])
#                     if j == len(prob_groups) - 1
#                     else max_prob
#                 )
#                 new_score = maxi / mini
#                 if new_score < current_score:
#                     best_swap = (i, j, k)
#                     current_score = new_score
#         # Consider taking something from b
#         for k, node_b in enumerate(group_b):
#             if not __all_compatible__(pcfg, node_b, group_a):
#                 continue
#             pb = node_b.probability
#             if prob + pb > max_prob:
#                 new_score = (prob + pb) / min_prob
#             else:
#                 new_score = max_prob / (prob + pb)
#             if new_score < current_score:
#                 best_swap = (i, None, k)
#                 current_score = new_score
#     return best_swap


# def __percolate_down__(prob_groups: List[List], group_index: int) -> None:
#     index = group_index
#     p = prob_groups[group_index][1]
#     while index > 0 and prob_groups[index - 1][1] > p:
#         prob_groups[index - 1], prob_groups[index] = (
#             prob_groups[index],
#             prob_groups[index - 1],
#         )
#         index -= 1


# def __percolate_up__(prob_groups: List[List], group_index: int) -> None:
#     index = group_index
#     p = prob_groups[group_index][1]
#     while index < len(prob_groups) - 2 and prob_groups[index + 1][1] < p:
#         prob_groups[index + 1], prob_groups[index] = (
#             prob_groups[index],
#             prob_groups[index + 1],
#         )
#         index += 1


# def __apply_swap__(
#     prob_groups: List[List], group_index: int, swap: Tuple[int, Optional[int], int]
# ) -> None:
#     j, k, l = swap
#     # App
#     if k:
#         node_a = prob_groups[group_index][0].pop(k)
#         prob_groups[group_index][1] -= node_a.probability
#         prob_groups[j][0].append(node_a)
#         prob_groups[j][1] += node_a.probability

#     node_b = prob_groups[j][0].pop(l)
#     prob_groups[j][1] -= node_b.probability
#     prob_groups[group_index][0].append(node_b)
#     prob_groups[group_index][1] += node_b.probability

#     __percolate_down__(prob_groups, -1)
#     __percolate_up__(prob_groups, group_index)


# def __split_into_nodes__(
#     pcfg: ProbDetGrammar[U, V, W], splits: int, desired_ratio: float = 2
# ) -> Tuple[List[List[Node]], float]:
#     nodes = __split_nodes_until_quantity_reached__(pcfg, splits)

#     # Create groups
#     groups: List[List[Node[U, W]]] = []
#     for node in nodes[:splits]:
#         groups.append([node])
#     for node in nodes[splits:]:
#         # Add to first compatible group
#         added = False
#         for group in groups:
#             if __all_compatible__(pcfg, node, group):
#                 group.append(node)
#                 added = True
#                 break
#         assert added

#     # Improve
#     improved = True
#     masses: List[float] = [np.sum([x.probability for x in group]) for group in groups]
#     prob_groups = sorted([[g, p] for g, p in zip(groups, masses)], key=lambda x: x[1])  # type: ignore
#     ratio: float = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
#     while improved and ratio > desired_ratio:
#         improved = False
#         for i in range(splits - 1):
#             swap = __find_swap_for_group__(pcfg, prob_groups, i)
#             if swap:
#                 improved = True
#                 __apply_swap__(prob_groups, i, swap)
#                 break
#         if not improved:
#             for i in range(splits - 1, 0, -1):
#                 improved = __try_split_node_in_group__(pcfg, prob_groups, i)
#                 if improved:
#                     break
#         ratio = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
#     return [g for g, _ in prob_groups], ratio  # type: ignore


# def split(
#     pcfg: ProbDetGrammar[U, V, W], splits: int, desired_ratio: float = 1.1
# ) -> Tuple[List[ProbDetGrammar[U, V, W]], float]:
#     """
#     Currently use exchange split.
#     Parameters:
#     desired_ratio: the max ratio authorized between the most probable group and the least probable pcfg

#     Return:
#     a list of ProbDetGrammar[U, V, W]
#     the reached threshold
#     """
#     if splits == 1:
#         return [pcfg], 1
#     assert desired_ratio > 1, "The desired ratio must be > 1!"
#     groups, ratio = __split_into_nodes__(pcfg, splits, desired_ratio)
#     return [__pcfg_from__(pcfg, group) for group in groups if len(group) > 0], ratio
