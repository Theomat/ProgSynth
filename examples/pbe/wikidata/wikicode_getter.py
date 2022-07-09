from typing import Dict, List, Tuple
from requests import get

# wikipedia pages fetcher
import wikipedia

ITEM = "item"
PROPERTY = "property"


def get_wikidata_json(text: str, object_type: str) -> Dict:
    """
    Although faster to use it without searching for wikipedia pages, way less safer and does not have suggestion possibility
    """
    response = get(
        "https://www.wikidata.org/w/api.php",
        {
            "action": "wbsearchentities",
            "language": "en",
            "type": object_type,
            "search": text,
            "format": "json",
            "limit": "10",
        },
    ).json()

    return response


def get_wikipedia_pages(text: str) -> str:
    print(wikipedia.search(text))
    page = wikipedia.page(text, auto_suggest=False)

    try:
        page = wikipedia.page(text)
    except wikipedia.exceptions.DisambiguationError as e:
        # returns the first (most probable) page when there is an ambiguation
        page = wikipedia.page(e.options[0])
    except wikipedia.exceptions.PageError:
        try:
            # try wikipedia suggestion
            suggestion = wikipedia.suggest(text)
            page = wikipedia.page(suggestion)
        except (wikipedia.exceptions.PageError, IndexError, ValueError):
            return (
                '[ERROR]: Item "' + text + '" does not match any pages. Try another id!'
            )
    print(page)
    return page.title


def get_q_code(text: str) -> Dict:
    """
    wrapper returning q code, for security purposes using wikipedia suggestion system
    """
    title = get_wikipedia_pages(text)
    if not title.startswith("[ERROR]"):
        response = get_wikidata_json(text=title, object_type=ITEM)
        return response
    else:
        print("Could not find the corresponding page.")
        return "[ERROR]"


def get_p_code(text: str) -> Dict:
    """
    same as get_q_code method, searching instead for p (property) codes
    Thus, cannot use wikipedia suggestions
    """
    return get_wikidata_json(text, PROPERTY)


if __name__ == "__main__":
    text = "New York City"
    data = get_wikidata_json(text, ITEM)
    print(data)
    code = data["search"][0]["id"]
    print(code)
