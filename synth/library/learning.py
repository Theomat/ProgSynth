from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

from synth.syntax.dsl import DSL
from synth.syntax.program import Function, Primitive, Program, Variable


_Graph = Tuple[
    Dict[int, Program],
    Dict[int, List[int]],
    Dict[Union[Primitive, Variable], List[int]],
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

    def copy(self) -> "_PartialTree":
        return _PartialTree(
            self.occurences[:],
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
        show = len(path) > 1  # or path[0] == 0
        if show:
            print("\t\t\t\t\tpath start >>>")
            print("\t\t\t\t\tobject:", graph[0][start])
            print("\t\t\t\t\tpath:", path)

        while i < len(path):
            index = path[-i - 1]
            local_edges = edges[start]
            if len(local_edges) <= index:
                return -1
            old_start = start
            start = edges[start][index]
            if show:
                print(
                    "\t\t\t\t\tpath:",
                    __prim__(graph[0][start]),
                    "edges:",
                    [__prim__(graph[0][x]) for x in edges[old_start]],
                )
                print(
                    "\t\t\t\t\tpath:",
                )
            i += 1
        if show:

            print("\t\t\t\t\tpath end <<<")

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
                        next = self.copy()
                        # Add link
                        path, program = next.add_link(graph, vertex, i, k)
                        # Check unique
                        r = next.unique_repr(graph, k)
                        if r not in done:
                            done.add(r)
                        else:
                            continue
                        # print("\t\t\tcurrent program:", self.string(graph))
                        # print("\t\t\ttotal program:", vertices[self.occurences[k]])

                        # print("\t\t\tpath:", path)
                        # print("\t\t\tadding:", program)

                        # Update occurences
                        next_occurences = [self.occurences[k]]
                        target_prim = __prim__(program)
                        for z in range(len(self.occurences)):
                            if z == k:
                                continue
                            real_vertex = next.follow_path_in_occurence(graph, z, path)
                            if (
                                real_vertex >= 0
                                and __prim__(vertices[real_vertex]) == target_prim
                            ):
                                next_occurences.append(self.occurences[z])

                        next.occurences = next_occurences
                        # print("\t\t\tnew program:", next.string(graph))

                        yield next

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
    vertices, edges, primitive2indices, vertex2size = graph
    P: Function = vertices[vertex]  # type: ignore
    # print("INIT:", P)
    occurences = primitive2indices[P.function]  # type: ignore
    # print("occurences:", occurences)
    max_size = max(vertex2size[v] for v in occurences)
    return _PartialTree(occurences, max_size, {0: [-1 for _ in edges[vertex]]}, {})


def __find_best__(
    graph: _Graph, best_score: int, done: Set[Tuple], tree: _PartialTree
) -> _PartialTree:
    if tree.max_score() <= best_score:
        return tree
    best = tree
    previous = tree.score() if tree.size() > 1 else -1
    # print("\tfind best from:", tree.string(graph))
    # print("\tstats: size:", tree.size(), "occ:", tree.num_occurences())
    local_best_score = previous
    for expansion in tree.expansions(graph, done):
        # print("\t\texpansion->", expansion.string(graph))
        # Use fact that score only increases then only decreases
        if expansion.score() <= previous:
            continue
        tree = __find_best__(graph, best_score, done, expansion)
        # print("\t\texpansion<-", tree.string(graph))
        if tree.score() > local_best_score:
            best = tree
    return best


def __programs_to_graph__(programs: List[Program]) -> _Graph:
    vertices: Dict[int, Program] = {}
    edges: Dict[int, List[int]] = {}
    primitive2indices: Dict[Union[Primitive, Variable], List[int]] = defaultdict(list)
    vertex2size: Dict[int, int] = {}
    for program in programs:
        args_indices: List[int] = []
        for el in program.depth_first_iter():
            i = len(vertices)
            vertices[i] = el
            vertex2size[i] = el.length()
            if isinstance(el, Function):
                primitive2indices[el.function].append(i)  # type: ignore
                args_len = len(el.function.type.arguments())
                edges[i] = args_indices[-args_len:]
                # Pop all consumed + the one for P.function which we did not consume
                args_indices = args_indices[: -(args_len + 1)]
            elif isinstance(el, (Primitive, Variable)):
                edges[i] = []
            args_indices.append(i)
        assert len(args_indices) == 1
    return vertices, edges, primitive2indices, vertex2size


def learn(programs: List[Program]) -> Tuple[int, int, str]:
    """
    Learn a new primitive from the specified benchmark

    Return:
        - new primitive size
        - number of occurences of the new primitive in the programs
        - str description of new primitive
    """

    done: Set[Tuple] = set()
    graph = __programs_to_graph__(programs)
    vertices = graph[0]
    best_score = 0
    best = None
    for vertex in range(len(vertices)):
        p = vertices[vertex]
        if not isinstance(p, Function):
            continue
        base_tree = __initial_tree__(graph, vertex)
        r = base_tree.unique_repr(graph, 0)
        if r in done:
            continue
        tree = __find_best__(graph, best_score, done, base_tree)
        done.add(r)
        if tree.score() > best_score:
            best_score = tree.score()
            best = tree
            print(f"[BEST] score={best_score} best={best.string(graph)}")
    if best is not None:
        print(best)
        print(best.string(graph))
        return best.size(), best.num_occurences(), best.string(graph)
    return 0, 0, ""


# def semantic_from_str(dsl: DSL, semantic: Dict[str, Any], desc: str) -> Callable:

#     pass
