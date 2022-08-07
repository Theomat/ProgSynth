import sys
from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON


def __make_query_path__(distance: int, id: int, tabs: int = 1) -> str:
    if distance == 0:
        return ("\t" * tabs) + f"w:{{}} ?p{distance} ?o_{id}_{distance} ."
    else:
        path = __make_query_path__(distance - 1, id, tabs) + "\n"
        path += (
            "\t" * tabs
        ) + f"?o_{id}_{distance-1} ?p{distance} ?o_{id}_{distance} ."
        return path


def __format__(el: str) -> str:
    return el.replace(" ", "_").replace("'", "_")


def build_search_path_query(entities: List[Tuple[str, str]], distance: int = 1) -> str:
    entities = [(__format__(a), __format__(b)) for a, b in entities]
    first = entities.pop()
    subquery = ""
    for i, item in enumerate(entities):
        subquery += "\tFILTER EXISTS {\n"
        subquery += (
            __make_query_path__(distance, i + 1, 2)
            .format(item[0])
            .replace(f"?o_{i + 1}_{distance}", "w:" + item[1])
        )
        subquery += "\n\t} .\n"
    sparql_request = "PREFIX w: <https://en.wikipedia.org/wiki/>\n"
    sparql_request += "SELECT "
    sparql_request += " ".join(f"?p{d}" for d in range(distance + 1))
    sparql_request += " WHERE {\n"
    sparql_request += (
        __make_query_path__(distance, 0)
        .format(first[0])
        .replace(f"?o_{0}_{distance}", "w:" + first[1])
    )
    sparql_request += "\n"
    sparql_request += subquery
    sparql_request += "\n}"
    return sparql_request


def build_count_paths_query(start: str, path: List[str]) -> str:
    sparql_request = "PREFIX w: <https://en.wikipedia.org/wiki/>\n"
    sparql_request += "SELECT "
    sparql_request += "?dst"
    sparql_request += " WHERE {\n"
    sparql_request += f"\tw:{__format__(start)} w:{path[0]} ?e0 ."
    for i in range(1, len(path) - 1):
        sparql_request += f"\t?e{i-1} w:{path[i]} ?e{i} ."
    sparql_request += f"\t?e{len(path) - 1} w:{path[-1]} ?dst"
    sparql_request += "\n}"
    return sparql_request


def build_wrapper(endpoint: str) -> SPARQLWrapper:
    wrapper = SPARQLWrapper(endpoint)
    wrapper.setReturnFormat(JSON)
    return wrapper


def __exec_search_path_query__(query: str, wrapper: SPARQLWrapper) -> List[List[str]]:
    if "+" in query or "|" in query:
        return []
    try:
        wrapper.setQuery(query)
        answer = wrapper.query().convert()
        paths: List[List[str]] = []
        for path in answer["results"]["bindings"]:
            cur_path = []
            for rel in path:
                cur_path.append(path[rel]["value"].split("/")[-1])
            paths.append(cur_path)
        return paths
    except Exception as e:
        print(e, file=sys.stderr)
        pass
    return []


def find_paths_from_level(
    pairs: List[Tuple[str, str]],
    wrapper: SPARQLWrapper,
    level: int,
    max_distance: int = 3,
) -> List[List[str]]:
    if level < 0:
        return []
    d = level
    while d < max_distance:
        query = build_search_path_query(pairs, d)
        out = __exec_search_path_query__(query, wrapper)
        if len(out) > 0:
            return out
        d += 1
    return []


def __exec_count_query__(query: str, wrapper: SPARQLWrapper) -> int:
    if "+" in query or "|" in query:
        return 0
    try:
        wrapper.setQuery(query)
        answer = wrapper.query().convert()
        return len(answer["results"]["bindings"])
    except Exception as e:
        print(e, file=sys.stderr)
        pass
    return 0


def choose_best_path(
    paths: List[List[str]], pairs: List[Tuple[str, str]], wrapper: SPARQLWrapper
) -> List[str]:
    best_path_index = 0
    best_score = 99999999999999999999
    for i, path in enumerate(paths):
        score = 0
        for start, _ in pairs:
            score += __exec_count_query__(build_count_paths_query(start, path), wrapper)
        if score < best_score:
            best_score = score
            best_path_index = i
    return paths[best_path_index]
