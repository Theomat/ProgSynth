from typing import List
from wrapper import ListWrapper
from wikicode_getter import get_wikidata_json, ITEM


"""
This file demonstrates a way to fetch from a list of str a wikimedia list object that contains, if the queries succeeded, the aforementioned list of str.
"""

def get_lists_url(
    items: List[str], num_lists: int = 1, num_list_per_property: int = 4
) -> List[List[str]]:
    q_codes: List[str] = []
    for item in items:
        data = get_wikidata_json(item, ITEM)
        code = "wd:" + data["search"][0]["id"]
        q_codes.append(code)
    wrapper = ListWrapper(q_codes)
    properties = wrapper.list_properties()
    num_properties = len(properties)
    lists: List[List[str]] = []
    for i in range(min(num_properties, num_lists)):
        lists.append(wrapper.get_lists(properties[i], num_list_per_property))
    return lists


if __name__ == "__main__":
    countries = ["France", "Poland", "Belgium", "Canada", "Djibouti"]
    urls = get_lists_url(countries)
