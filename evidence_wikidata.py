import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "DSC-Hackathon-Hallucination-Detector/1.0 (contact: student-project)"
}

def safe_get_json(url, params=None):
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            return None

        return response.json()
    except:
        return None


def query_wikidata_capital(country):
    """
    Returns capital of a country using Wikidata.
    Fails safely (returns None).
    """

    search_params = {
        "action": "wbsearchentities",
        "search": country,
        "language": "en",
        "format": "json"
    }

    search_data = safe_get_json(WIKIDATA_API, search_params)
    if not search_data or not search_data.get("search"):
        return None

    country_id = search_data["search"][0]["id"]

    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{country_id}.json"
    entity_data = safe_get_json(entity_url)

    if not entity_data:
        return None

    claims = entity_data["entities"][country_id].get("claims", {})

    # P36 = capital
    if "P36" not in claims:
        return None

    capital_id = claims["P36"][0]["mainsnak"]["datavalue"]["value"]["id"]

    capital_entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{capital_id}.json"
    capital_data = safe_get_json(capital_entity_url)

    if not capital_data:
        return None

    return capital_data["entities"][capital_id]["labels"]["en"]["value"]

def search_entity_id(name):

    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json"
    }

    data = safe_get_json(
        WIKIDATA_API,
        params
    )

    if not data:
        return None

    if not data.get("search"):
        return None

    return data["search"][0]["id"]

def get_entity_claim(entity_id, property_id):

    entity_url = (
        f"https://www.wikidata.org/wiki/"
        f"Special:EntityData/{entity_id}.json"
    )

    data = safe_get_json(entity_url)

    if not data:
        return None

    claims = (
        data["entities"][entity_id]
        .get("claims", {})
    )

    if property_id not in claims:
        return None

    try:

        target_id = (
            claims[property_id][0]
            ["mainsnak"]
            ["datavalue"]
            ["value"]
            ["id"]
        )

        return target_id

    except:
        return None
    
def get_entity_label(entity_id):

    entity_url = (
        f"https://www.wikidata.org/wiki/"
        f"Special:EntityData/{entity_id}.json"
    )

    data = safe_get_json(entity_url)

    if not data:
        return None

    return (
        data["entities"][entity_id]
        ["labels"]["en"]["value"]
    )

def query_wikidata_deathplace(person):

    person_id = search_entity_id(person)

    if not person_id:
        return None

    # P20 = place of death

    place_id = get_entity_claim(
        person_id,
        "P20"
    )

    if not place_id:
        return None

    return get_entity_label(place_id)
