from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://192.168.1.20:9999/blazegraph/namespace/kb/sparql"


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


def build_query(entities: List[Tuple[str, str]], distance: int = 1) -> str:
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


def execute_query(query: str, endpoint: str = SPARQL_ENDPOINT) -> List[List[str]]:
    wrapper = SPARQLWrapper(endpoint)
    wrapper.setReturnFormat(JSON)
    wrapper.setQuery(query)
    answer = wrapper.query().convert()
    paths: List[List[str]] = []
    for path in answer["results"]["bindings"]:
        cur_path = []
        for rel in path:
            cur_path.append(path[rel]["value"].split("/")[-1])
        paths.append(cur_path)
    return paths


if __name__ == "__main__":
    countries = [
        ("France", "Paris"),
        ("United_States", "Washington"),
        ("Germany", "Berlin"),
    ]
    query = build_query(countries, 2)
    print(query)
    print(execute_query(query))
