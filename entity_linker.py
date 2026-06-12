ENTITY_ALIASES = {

    "usa": "united states",
    "us": "united states",
    "america": "united states",
    "u.s.a.": "united states",

    "uk": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",

    "uae": "united arab emirates",

    "nyc": "new york city",

    "delhi": "new delhi"
}


def canonicalize_entity(entity):

    if not entity:
        return entity

    entity = entity.lower().strip()

    return ENTITY_ALIASES.get(
        entity,
        entity
    )