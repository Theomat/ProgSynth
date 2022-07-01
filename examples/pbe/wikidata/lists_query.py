from typing import Dict, List
from wrapper import ListWrapper
from wikicode_getter import get_wikidata_json, ITEM
import requests as r
import pandas as pd

"""
This file demonstrates a way to fetch from a list of str a wikimedia list object that contains, if the queries succeeded, the aforementioned list of str.
"""


def get_lists_url(
    items: List[str], num_lists: int = 2, num_list_per_property: int = 2
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


"""
this method will use pandas in order to generate .csv files of the tables extracted from html wikipedia pages.
urls: the list of wikipedia page of the list
match: a string that must be contained in the table extracted by pandas.
"""


def get_lists(urls: List[List[str]], match: str) -> None:
    for list in urls:
        for url in list:
            try:
                response = r.get(url)
                json = response.json()
                # there may be several entities in one response
                for field in json["entities"]:
                    wikien_url = json["entities"][field]["sitelinks"]["enwiki"]["url"]
                    name = wikien_url.split("/")[-1] + ".csv"
                    html = r.get(wikien_url).content
                    df_list = pd.read_html(html, match=match)
                    df = df_list[-1]
                    df.to_csv(name)
            except Exception as e:
                print("Exception : ", e)


if __name__ == "__main__":
    countries = ["France", "Poland", "Belgium", "Canada", "Djibouti"]
    urls = get_lists_url(countries)
    get_lists(urls, "France")
