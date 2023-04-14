from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import tqdm

from synth.syntax.program import Function, Primitive, Program, Variable


_Graph = Tuple[
    Dict[int, Program],
    Dict[int, List[int]],
    Dict[Union[Primitive, Variable], List[int]],
    Dict[int, int],
    Dict[int, int],
]


def __prim__(p: Program) -> Program:
    if isinstance(p, Function):
        return p.function
    else:
        return p


@dataclass
class _PartialTree:
    occurences: List[int]
    occurences_vertices: List[Set[int]]
    max_size: int
    structure: Dict[int, List[int]]
    parents: Dict[int, Tuple[int, int]]

    def score(self) -> int:
        return self.size() * self.num_occurences()

    def max_score(self) -> int:
        return self.max_size * self.num_occurences()

    def num_occurences(self) -> int:
        return len(self.occurences)

    def size(self) -> int:
        return len(self.structure)

    def partial_copy(self) -> "_PartialTree":
        return _PartialTree(
            self.occurences[:],
            [],
            self.max_size,
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
        edges = graph[1]
        start = self.occurences[occurence_index]
        i = 0
        while i < len(path):
            index = path[-i - 1]
            local_edges = edges[start]
            if len(local_edges) <= index:
                return -1
            start = edges[start][index]
            i += 1
        return start

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
        vertex2tree = graph[-1]
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
    vertices, edges, primitive2indices, vertex2size, _ = graph
    P: Function = vertices[vertex]  # type: ignore
    occurences = primitive2indices[P.function]  # type: ignore
    occurences_vertices: List[Set[int]] = []
    max_size: int = 0
    for v in occurences:
        s = vertex2size[v]
        if s > max_size:
            max_size = s
        occurences_vertices.append({v})
    return _PartialTree(
        occurences,
        occurences_vertices,
        max_size,
        {0: [-1 for _ in edges[vertex]]},
        {},
    )


def __find_best__(
    graph: _Graph, best_score: int, done: Set[Tuple], tree: _PartialTree
) -> _PartialTree:
    if tree.max_score() <= best_score:
        return tree
    best = tree
    previous = tree.score() if tree.size() > 1 else -1
    local_best_score = previous
    for expansion in tree.expansions(graph, done):
        # Use fact that score only increases then only decreases
        if expansion.score() <= previous:
            continue
        tree = __find_best__(graph, best_score, done, expansion)
        if tree.score() > local_best_score:
            best = tree
    return best


def __programs_to_graph__(programs: List[Program]) -> _Graph:
    vertices: Dict[int, Program] = {}
    edges: Dict[int, List[int]] = {}
    primitive2indices: Dict[Union[Primitive, Variable], List[int]] = defaultdict(list)
    vertex2size: Dict[int, int] = {}
    vertex2tree: Dict[int, int] = {}
    for tree_no, program in enumerate(programs):
        args_indices: List[int] = []
        for el in program.depth_first_iter():
            vertex = len(vertices)
            vertices[vertex] = el
            vertex2tree[vertex] = tree_no
            vertex2size[vertex] = el.size()
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
    return vertices, edges, primitive2indices, vertex2size, vertex2tree


def learn(programs: List[Program], progress: bool = False) -> Tuple[int, int, str]:
    """
    Learn a new primitive from the specified benchmark

    Return:
        - new primitive size
        - number of occurences of the new primitive in the programs
        - str description of new primitive
    """

    done: Set[Tuple] = set()
    done_primitives: Set[Program] = set()
    graph = __programs_to_graph__(programs)
    vertices = graph[0]
    best_score = 0
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
        tree = __find_best__(graph, best_score, done, base_tree)
        done.add(r)
        if tree.score() > best_score:
            best_score = tree.score()
            best = tree
            if pbar:
                pbar.set_postfix_str(f"{best.string(graph)} ({best_score})")
    if best is not None:
        return best.size(), best.num_occurences(), best.string(graph)
    return 0, 0, ""
