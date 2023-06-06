from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import tqdm

from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import Type

import numpy as np


_Graph = Tuple[
    Dict[int, Program],
    Dict[int, List[int]],
    Dict[Union[Primitive, Variable], List[int]],
    Dict[int, int],
    Dict[int, int],
]


def __prim__(p: Program) -> Primitive:
    if isinstance(p, Function):
        return p.function  # type: ignore
    else:
        return p  # type: ignore


@dataclass
class _PartialTree:
    occurences: List[int]
    occurences_vertices: List[Set[int]]
    structure: Dict[int, List[int]]
    parents: Dict[int, Tuple[int, int]]

    def num_occurences(self) -> int:
        return len(self.occurences)

    def size(self) -> int:
        return len(self.structure)

    def partial_copy(self) -> "_PartialTree":
        return _PartialTree(
            self.occurences[:],
            [],
            {k: v[:] for k, v in self.structure.items()},
            {k: (v[0], v[1]) for k, v in self.parents.items()},
        )

    def unique_repr(self, graph: _Graph, occurence_index: int) -> Tuple:
        """
        Compute a unique hashable representation for set membership queries
        """
        vertices, edges = graph[0], graph[1]
        start = self.occurences[occurence_index]
        todo: List[Tuple[Optional[int], Optional[int]]] = [(start, 0)]
        out: Tuple = (None,)
        while todo:
            real_vertex, local_vertex = todo.pop()
            if real_vertex is None or local_vertex is None:
                out = (None, out)
                continue
            out = (str(__prim__(vertices[real_vertex])), out)
            for i, local in enumerate(self.structure[local_vertex]):
                if local >= 0:
                    todo.append((edges[real_vertex][i], local))
                else:
                    todo.append((None, None))
        return out

    def path(self, local_vertex: int) -> List[int]:
        path = []
        current = local_vertex
        while current != 0:
            parent, index = self.parents[current]
            path.append(index)
            current = parent
        return path

    def follow_path_in_occurence(
        self, graph: _Graph, occurence_index: int, path: List[int]
    ) -> int:
        """
        Compute vertex index in occurence in graph following the path from the start
        """
        return self.follow_path(graph, self.occurences[occurence_index], path)

    def follow_path(self, graph: _Graph, start: int, path: List[int]) -> int:
        """
        Compute global end vertex from start following path
        """
        edges = graph[1]
        i = 0
        while i < len(path):
            index = path[-i - 1]
            local_edges = edges[start]
            if len(local_edges) <= index:
                return -1
            start = edges[start][index]
            i += 1
        return start

    def all_vertices_for_occurence(
        self, graph: _Graph, occurence_index: int
    ) -> Set[int]:
        edges = graph[1]
        out: Set[int] = set()
        stack: List[Tuple[int, int]] = [(self.occurences[occurence_index], 0)]
        while stack:
            glbl, lcl = stack.pop()
            out.add(glbl)
            outgoing = self.structure.get(lcl, [])
            for i, el in enumerate(outgoing):
                local_edges = edges[glbl]
                if len(local_edges) <= i:
                    break
                stack.append((local_edges[i], el))

        return out

    def add_link(
        self,
        graph: _Graph,
        local_parent: int,
        local_child_no: int,
        occurence_index: int,
    ) -> Tuple[List[int], Program]:
        # Add in structure
        j = len(self.structure)
        self.structure[local_parent][local_child_no] = j
        self.parents[j] = (local_parent, local_child_no)
        # Match with reality
        path = self.path(j)
        new_real_vertex = self.follow_path_in_occurence(graph, occurence_index, path)
        if new_real_vertex == -1:
            return [], graph[0][0]
        vertices = graph[0]
        program = vertices[new_real_vertex]
        # Finish structure
        self.structure[j] = [-1 for _ in program.type.arguments()]
        return path, program

    def expansions(
        self, graph: _Graph, done: Set[Tuple]
    ) -> Generator["_PartialTree", None, None]:
        vertices = graph[0]
        for vertex, edges in self.structure.items():
            for i, edge in enumerate(edges):
                if edge < 0:
                    for k in range(len(self.occurences)):
                        next = self.partial_copy()
                        # Add link
                        path, program = next.add_link(graph, vertex, i, k)
                        # If path is empty we have a type mismatch
                        if len(path) == 0:
                            continue
                        # Check unique
                        r = next.unique_repr(graph, k)
                        if r not in done:
                            done.add(r)
                        else:
                            continue

                        # Update occurences
                        next_occurences = []
                        to_add = []
                        target_prim = __prim__(program)
                        for z in range(len(self.occurences)):
                            real_vertex = next.follow_path_in_occurence(graph, z, path)
                            if (
                                real_vertex >= 0
                                and __prim__(vertices[real_vertex]) == target_prim
                            ):
                                next_occurences.append(self.occurences[z])
                                next.occurences_vertices.append(
                                    self.occurences_vertices[z].copy()
                                )
                                to_add.append(real_vertex)

                        next.occurences = next_occurences

                        for x in next.__disambiguity__(graph, to_add):
                            yield x

    def __disambiguity__(
        self, graph: _Graph, to_add: List[int], start: int = 0
    ) -> Generator["_PartialTree", None, None]:
        vertex2tree = graph[-2]
        # Check for intersection between intersections
        inside = [True for _ in self.occurences]
        for i in range(start, len(self.occurences)):
            if not inside[i]:
                continue
            tree_i = vertex2tree[self.occurences[i]]
            for j in range(i + 1, len(self.occurences)):
                if not inside[j]:
                    continue
                if vertex2tree[self.occurences[j]] != tree_i:
                    continue
                if (
                    to_add[i] in self.occurences_vertices[j]
                    or to_add[j] in self.occurences_vertices[j]
                ):
                    other = self.partial_copy()
                    other.occurences = [
                        x
                        for z, x in enumerate(other.occurences)
                        if inside[z] and z != i
                    ]
                    other.occurences_vertices = [
                        x
                        for z, x in enumerate(self.occurences_vertices)
                        if inside[z] and z != i
                    ]
                    for x in other.__disambiguity__(graph, to_add, i + 1):
                        yield x
                    inside[j] = False
        # Finally add elements
        for i in range(len(self.occurences)):
            if inside[i]:
                self.occurences_vertices[i].add(to_add[i])
        # Filter
        self.occurences = [x for z, x in enumerate(self.occurences) if inside[z]]
        self.occurences_vertices = [
            x for z, x in enumerate(self.occurences_vertices) if inside[z]
        ]

        yield self

    def string(self, graph: _Graph) -> str:
        vertices, edges = graph[0], graph[1]
        out = ""
        todo: List[Tuple[Optional[int], Optional[int]]] = [(self.occurences[0], 0)]
        close_parenthesis: List[int] = []
        while todo:
            real, current = todo.pop(0)
            if current is None or real is None:
                out += "_"
                close_parenthesis[-1] -= 1
                if close_parenthesis[-1] == 0:
                    out += ")"
                else:
                    out += " "
                continue
            fun = any(arg >= 0 for arg in self.structure[current])
            if fun:
                out += "("
                close_parenthesis.append(len(self.structure[current]) + 1)
                for i, arg in enumerate(self.structure[current]):
                    if arg >= 0:
                        todo.insert(i, (edges[real][i], arg))
                    else:
                        todo.insert(i, (None, None))
            out += str(__prim__(vertices[real]))
            if close_parenthesis:
                close_parenthesis[-1] -= 1
                if close_parenthesis[-1] == 0:
                    out += ")"
                else:
                    out += " "
            else:
                out += " "

        return out


def __initial_tree__(graph: _Graph, vertex: int) -> _PartialTree:
    vertices, edges, primitive2indices, _, __ = graph
    P: Function = vertices[vertex]  # type: ignore
    occurences = primitive2indices[P.function]  # type: ignore
    occurences_vertices: List[Set[int]] = [{v} for v in occurences]
    return _PartialTree(
        occurences,
        occurences_vertices,
        {0: [-1 for _ in edges[vertex]]},
        {},
    )


def __find_best__(
    graph: _Graph,
    best_score: float,
    done: Set[Tuple],
    tree: _PartialTree,
    score_function: Callable[[_Graph, _PartialTree], float],
) -> _PartialTree:
    best = tree
    previous = score_function(graph, tree) if tree.size() > 1 else -float("inf")
    local_best_score = previous
    for expansion in tree.expansions(graph, done):
        # Use fact that score only increases then only decreases
        if score_function(graph, expansion) <= previous:
            continue
        tree = __find_best__(graph, best_score, done, expansion, score_function)
        if score_function(graph, tree) > local_best_score:
            best = tree
    return best


def __programs_to_graph__(programs: List[Program]) -> _Graph:
    vertices: Dict[int, Program] = {}
    edges: Dict[int, List[int]] = {}
    primitive2indices: Dict[Union[Primitive, Variable], List[int]] = defaultdict(list)
    vertex2tree: Dict[int, int] = {}
    tree2start: Dict[int, int] = {}

    for tree_no, program in enumerate(programs):
        tree2start[tree_no] = len(vertices)
        args_indices: List[int] = []
        for el in program.depth_first_iter():
            vertex = len(vertices)
            vertices[vertex] = el
            vertex2tree[vertex] = tree_no
            if isinstance(el, Function):
                primitive2indices[el.function].append(vertex)  # type: ignore
                args_len = len(el.function.type.arguments()) - len(el.type.arguments())
                edges[vertex] = args_indices[-args_len:]
                # Pop all consumed + the one for P.function which we did not consume
                args_indices = args_indices[: -(args_len + 1)]
            elif isinstance(el, (Primitive, Variable)):
                edges[vertex] = []
            args_indices.append(vertex)
        assert len(args_indices) == 1, f"args_indices:{args_indices}"
    return vertices, edges, primitive2indices, vertex2tree, tree2start


def score_description(graph: _Graph, tree: _PartialTree) -> float:
    return tree.num_occurences() * tree.size()


def make_score_probabilistic(
    programs: List[Program], predict_vars: bool = True, var_prob: float = 0.2
) -> Callable[[_Graph, _PartialTree], float]:
    type2dict: Dict[Type, Dict[Tuple[str, str, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    tree2program = {}
    for tree_no, program in enumerate(programs):
        tree2program[tree_no] = program
        type2count = type2dict[program.type]
        for P in program.depth_first_iter():
            if isinstance(P, Function):
                primitive = __prim__(P).primitive
                for arg_no, arg in enumerate(P.arguments):
                    type2count[(primitive, str(__prim__(arg)), arg_no)] += 1

    lvar_prob = np.log(var_prob)

    def probability(
        cur_type2dict: Dict[Type, Dict[Tuple[str, str, int], int]]
    ) -> float:
        total: Dict[Type, Dict[Tuple[str, int], int]] = {}
        vars = defaultdict(set)
        for t in cur_type2dict:
            total[t] = defaultdict(int)
            for (a, b, c), count in cur_type2dict[t].items():
                if not predict_vars and b.startswith("var"):
                    vars[t].add((a, c))
                    continue
                total[t][(a, c)] += count

        normed = {
            t: {
                (a, b, c): np.log(count / total[t][(a, c)])
                for (a, b, c), count in cur_type2dict[t].items()
                if count > 0 and total[t][(a, c)] > 0
            }
            for t in cur_type2dict
        }
        prob = 0
        for t in cur_type2dict:
            for (a, b, c), p in normed[t].items():
                if not predict_vars and b.startswith("var"):
                    p = lvar_prob if total[t][(a, c)] > 0 else 0
                count = cur_type2dict[t][(a, b, c)]
                prob += p * count
                # print("key:", (a, b, c), "prob:", p, "*", count)

        return prob

    # print("ORIGINAL:", probability(type2dict))
    SPECIAL_STRING = "zefjpozjqfpokqzùofkepqozkfùpokqzefjùqzifjeùpoqzefùpoqkfeokqzofkùezùqofkeùozqkfoe"

    def score(graph: _Graph, tree: _PartialTree) -> float:
        cur_type2dict = {
            k: {kk: vv for kk, vv in v.items()} for k, v in type2dict.items()
        }
        vertices, edges, primitive2indices, vertex2tree, tree2start = graph
        # print("TREE:", tree.string(graph))
        for k in range(len(tree.occurences)):
            start_of_occ = tree.occurences[k]
            tree_no = vertex2tree[start_of_occ]
            program: Program = tree2program[tree_no]
            type2count = cur_type2dict[program.type]
            vertex = tree2start[tree_no]
            args_indices: List[Tuple[int, bool]] = []
            belonging = tree.all_vertices_for_occurence(graph, k)
            for el in program.depth_first_iter():
                belongs = vertex in belonging
                if isinstance(el, Function):
                    args_len = len(el.arguments)
                    primitive = __prim__(el).primitive
                    my_args = args_indices[-args_len:]
                    # print("\t", el.arguments, my_args)
                    # print("\tcurrent:", el)
                    if belongs:
                        for arg_no in range(args_len):
                            arg = el.arguments[arg_no]
                            arg_s = str(__prim__(arg))
                            if my_args[arg_no][1]:
                                key = (primitive, arg_s, arg_no)
                                # if key in type2count:
                                type2count[(primitive, arg_s, arg_no)] -= 1
                            else:
                                key = (SPECIAL_STRING, arg_s, arg_no)
                                if key not in type2count:
                                    type2count[key] = 0
                                type2count[key] += 1
                    elif not belongs and any(b for _, b in my_args):
                        for i, (_, b) in enumerate(my_args):
                            if b:
                                key = (primitive, SPECIAL_STRING, i)
                                if key not in type2count:
                                    type2count[key] = 0
                                type2count[key] += 1
                    # Pop all consumed + the one for P.function which we did not consume
                    args_indices = args_indices[: -(args_len + 1)]
                args_indices.append((vertex, belongs))
                vertex += 1

        return probability(cur_type2dict)

    return score


def learn(
    programs: List[Program],
    score_function: Callable[[_Graph, _PartialTree], float] = score_description,
    progress: bool = False,
) -> Tuple[float, str]:
    """
    Learn a new primitive from the specified benchmark.
    The default scoring function maximise the gain in description size of the dataset.
    Suppose:
        for all x, score_function(x) > -inf
        score_function is computed in the worst case in polynomial time

    Return:
        - cost of new primitive
        - str description of new primitive
    """

    done: Set[Tuple] = set()
    done_primitives: Set[Program] = set()
    graph = __programs_to_graph__(programs)
    vertices = graph[0]
    best_score = -float("inf")
    best = None
    pbar = None
    if progress:
        pbar = tqdm.tqdm(total=len(vertices))
    for vertex in range(len(vertices)):
        if pbar:
            pbar.update(1)
        p = vertices[vertex]
        if not isinstance(p, Function):
            continue
        if __prim__(p) in done_primitives:
            continue
        done_primitives.add(__prim__(p))
        base_tree = __initial_tree__(graph, vertex)
        r = base_tree.unique_repr(graph, 0)
        tree = __find_best__(graph, best_score, done, base_tree, score_function)
        done.add(r)
        ts = score_function(graph, tree)
        if ts > best_score:
            best_score = ts
            best = tree
            if pbar:
                pbar.set_postfix_str(f"{best.string(graph)} ({best_score})")
    if best is not None:
        return best_score, best.string(graph)
    return 0.0, ""
