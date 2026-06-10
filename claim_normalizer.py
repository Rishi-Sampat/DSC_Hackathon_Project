import re


def normalize_claim(text: str) -> dict:

    text = text.strip().lower()

    # ==========================================
    # CAPITAL RELATION
    # ==========================================
    match = re.search(r"(.*?) is the capital of (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "capital_of",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # COUNT RELATION
    # ==========================================
    match = re.search(r"(.*?) has (\d+) (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "count",
            "subject": match.group(1).title(),
            "object": match.group(3).replace("s", "").replace("ies", "y"),
            "value": int(match.group(2))
        }

    # ==========================================
    # LOCATION RELATION
    # ==========================================
    match = re.search(r"(.*?) is in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "located_in",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # BORN IN
    # ==========================================
    match = re.search(r"(.*?) was born in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "born_in",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # DIED IN
    # ==========================================
    match = re.search(r"(.*?) died in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "died_in",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # INVENTED BY
    # ==========================================
    match = re.search(r"(.*?) invented (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "invented_by",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # OCCUPATION
    # Einstein was a physicist
    # ==========================================
    match = re.search(r"(.*?) was a[n]? (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "occupation",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # NATIONALITY
    # Hitler was British
    # Einstein was German
    # ==========================================
    match = re.search(r"(.*?) was (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "nationality",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # IS-A RELATION
    # Kangaroo is a marsupial
    # ==========================================
    match = re.search(r"(.*?) is a[n]? (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "is_a",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # ==========================================
    # FALLBACK
    # ==========================================
    return {
        "type": "unstructured",
        "relation": None,
        "subject": text,
        "object": None,
        "value": None
    }