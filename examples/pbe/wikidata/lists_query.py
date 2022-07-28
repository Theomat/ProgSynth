from typing import List, Set, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://192.168.1.20:9999/blazegraph/namespace/kb/sparql"


def get_path(entities: List[Tuple[str, str]]) -> None:
    wrapper = SPARQLWrapper(SPARQL_ENDPOINT)
    wrapper.setReturnFormat(JSON)

    first = entities.pop()
    subquery = ""
    for i, item in enumerate(entities):
        subquery += f"\tFILTER EXISTS {{ yago:{item[0]} ?p1 ?o{i} . \n\t\t\t ?o{i} ?p2 yago:{item[1]}}} .\n"
    sparql_request = f"""PREFIX yago: <http://yago-knowledge.org/resource/>
    SELECT ?p1 ?p2 ?obj
    WHERE
    {{
        yago:{first[0]} ?p1 ?obj .
        ?obj ?p2 yago:{first[1]}
{subquery}
    }}"""
    wrapper.setQuery(sparql_request)
    answer = wrapper.query().convert()
    print(sparql_request)
    print()
    print(answer)

    properties: Set[str] = set()
    for r in answer["results"]["bindings"]:
        properties.add("yago:" + r["obj"]["value"].split("/")[-1])
    return properties


if __name__ == "__main__":
    countries = [("France", "Paris"), ("Poland", "Warsaw"), ("Belgium", "Brussels")]
    print(get_path(countries))
