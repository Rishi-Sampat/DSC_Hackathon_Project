import re

def normalize_claim(text: str) -> dict:
    text = text.strip().lower()

    # -----------------------------
    # CAPITAL RELATION
    # -----------------------------
    match = re.search(r"(.*?) is the capital of (.*)", text)
    if match:
        return {
            "type": "structured",
            "relation": "capital_of",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # -----------------------------
    # COUNT RELATION (numbers)
    # -----------------------------
    match = re.search(r"(.*?) has (\d+) (.*)", text)
    if match:
        return {
            "type": "structured",
            "relation": "count",
            "subject": match.group(1).title(),
            "object": match.group(3).replace("s", "").replace("ies", "y"),
            "value": int(match.group(2))
        }

    # -----------------------------
    # LOCATION RELATION
    # -----------------------------
    match = re.search(r"(.*?) is in (.*)", text)
    if match:
        return {
            "type": "structured",
            "relation": "located_in",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # -----------------------------
    # INVENTION RELATION
    # -----------------------------
    match = re.search(r"(.*?) invented (.*)", text)
    if match:
        return {
            "type": "structured",
            "relation": "invented_by",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # -----------------------------
    # IS-A RELATION (fallback structure)
    # -----------------------------
    match = re.search(r"(.*?) is a (.*)", text)
    if match:
        return {
            "type": "structured",
            "relation": "is_a",
            "subject": match.group(1).title(),
            "object": match.group(2).title(),
            "value": None
        }

    # -----------------------------
    # FALLBACK
    # -----------------------------
    return {
        "type": "unstructured",
        "relation": None,
        "subject": text,
        "object": None,
        "value": None
    }
