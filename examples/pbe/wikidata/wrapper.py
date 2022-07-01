from typing import List
from SPARQLWrapper import SPARQLWrapper, JSON


class Wrapper:
    def __init__(self, url: str, query: str) -> None:
        self._wrapper = SPARQLWrapper(url)
        self._query = query

    def query(self) -> str:
        """
        queries to the wrapped url the query content
        """
        self._wrapper.setQuery(self._query)
        self._wrapper.setReturnFormat(JSON)
        results = self._wrapper.query().convert()

        return results


class ListWrapper(Wrapper):
    objects: List[str] = []

    def __init__(
        self,
        objects: List[str],
        url: str = "https://query.wikidata.org/sparql",
        query: str = "",
    ) -> None:
        super().__init__(url, query)
        self.objects = objects.copy()

    def set_objects(self, obj: List[str]) -> None:
        self.objects = obj.copy()

    # will list all properties common between each item of the list.
    def list_properties(self) -> List[str]:
        if len(self.objects) < 1:
            print("No objects represented by wikidata q-codes received.")
            return []

        first = self.objects[0]
        subquery = ""
        for item in self.objects[1:]:
            subquery += f"FILTER EXISTS {{ {item} ?property ?obj .}} .\n"
        self._query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?property ?obj 
WHERE
{{
    {first} ?property ?obj .
    ?prop wikibase:directClaim ?property .
    {subquery}
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,en" . }}
}}
    """
        answer = self.query()
        # expected answer: uri pages ending with searched qcode
        properties: List[str] = []
        for r in answer["results"]["bindings"]:
            properties.append("wd:" + r["obj"]["value"].split("/")[-1])
        return properties

    # this method is expected to be used after list_properties.
    # will return a number of lists (if any existing) with as a subject the given property.
    def get_lists(self, property: str, max_lists=5) -> List[str]:
        self._query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?list 
WHERE
{{
  ?list wdt:P31 wd:Q13406463 .
  ?list ?property {property} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,en" . }}
}}
        """
        answer = self.query()
        lists: List[str] = []
        num_lists = 0
        for r in answer["results"]["bindings"]:
            if num_lists >= max_lists:
                break
            num_lists += 1
            lists.append(r["list"]["value"])
        return lists
