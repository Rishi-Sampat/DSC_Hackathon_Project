# ------------------------------------
# ENTITY ALIASES
# ------------------------------------

ENTITY_ALIASES = {

    # Countries
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",

    "uk": "united kingdom",
    "u.k.": "united kingdom",

    "uae": "united arab emirates",

    # Capitals / Cities
    "delhi": "new delhi",

    # Country names
    "bharat": "india",

    # Optional common names
    "russia": "russian federation"
}


def normalize_entity(entity: str) -> str:
    """
    Normalize entity names before comparison.
    """

    if not entity:
        return ""

    entity = entity.strip().lower()

    return ENTITY_ALIASES.get(entity, entity)


def entities_match(entity1: str, entity2: str) -> bool:
    """
    Checks if two entities should be considered equivalent.
    """

    e1 = normalize_entity(entity1)
    e2 = normalize_entity(entity2)

    if e1 == e2:
        return True

    # Partial containment
    if e1 in e2 or e2 in e1:
        return True

    return False