from typing import List, Set, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://192.168.1.20:9999/blazegraph/namespace/kb/sparql"


def make_query_path(distance: int, id: int, tabs: int = 1) -> str:
    if distance == 0:
        return ("\t" * tabs) + f"yago:{{}} ?p{distance} ?o_{id}_{distance} ."
    else:
        path = make_query_path(distance - 1, id, tabs) + "\n"
        path += (
            "\t" * tabs
        ) + f"?o_{id}_{distance-1} ?p{distance} ?o_{id}_{distance} ."
        return path


def build_query(entities: List[Tuple[str, str]], distance: int = 1) -> str:
    first = entities.pop()
    subquery = ""
    for i, item in enumerate(entities):
        subquery += "\tFILTER EXISTS {\n"
        subquery += (
            make_query_path(distance, i + 1, 2)
            .format(item[0])
            .replace(f"?o_{i + 1}_{distance}", "yago:" + item[1])
        )
        subquery += "\n\t} .\n"
    sparql_request = "PREFIX yago: <http://yago-knowledge.org/resource/>\n"
    sparql_request += "SELECT "
    sparql_request += " ".join(f"?p{d}" for d in range(distance + 1))
    sparql_request += " WHERE {\n"
    sparql_request += (
        make_query_path(distance, 0)
        .format(first[0])
        .replace(f"?o_{0}_{distance}", "yago:" + first[1])
    )
    sparql_request += "\n"
    sparql_request += subquery
    sparql_request += "\n}"
    return sparql_request


def execute_query(query: str, endpoint: str = SPARQL_ENDPOINT) -> Set[str]:
    wrapper = SPARQLWrapper(endpoint)
    wrapper.setReturnFormat(JSON)
    wrapper.setQuery(query)
    answer = wrapper.query().convert()
    properties: Set[str] = set()
    for r in answer["results"]["bindings"]:
        properties.add("yago:" + r["obj"]["value"].split("/")[-1])
    return properties


if __name__ == "__main__":
    countries = [("France", "Paris"), ("Poland", "Warsaw"), ("Belgium", "Brussels")]
    query = build_query(countries, 2)
    print(query)
    print(execute_query(query))
