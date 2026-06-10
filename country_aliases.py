COUNTRY_ALIASES = {

    "germany": ["german"],
    "india": ["indian"],
    "france": ["french"],
    "britain": ["british"],
    "united kingdom": ["british"],
    "america": ["american"],
    "united states": ["american"],
    "china": ["chinese"],
    "japan": ["japanese"],
    "italy": ["italian"],
    "russia": ["russian"],
    "canada": ["canadian"],
    "australia": ["australian"],
    "spain": ["spanish"],
    "switzerland": ["swiss"],
    "austria": ["austrian"]
}


def country_match(country, evidence):
    """
    Checks if a country or its nationality form
    appears in evidence text.
    """

    country = country.lower()
    evidence = evidence.lower()

    if country in evidence:
        return True

    aliases = COUNTRY_ALIASES.get(country, [])

    for alias in aliases:
        if alias in evidence:
            return True

    return False