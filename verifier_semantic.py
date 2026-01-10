from evidence_wikidata import query_wikidata_capital
from evidence_wikipedia import query_wikipedia_summary


def verify_structured_claim(claim):
    """
    Verifies structured claims using Wikidata (primary) + Wikipedia (fallback).
    Returns: (truth_status, sources)
    """

    relation = claim["relation"]

    # ---------------- CAPITAL ----------------
    if relation == "capital_of":
        subject = claim["subject"]
        obj = claim["object"]

        # Try Wikidata first
        capital = query_wikidata_capital(obj)

        if capital:
            if capital.lower() == subject.lower():
                wiki = query_wikipedia_summary(f"Capital of {obj}")
                return "True", [wiki] if wiki else []

            wiki = query_wikipedia_summary(subject)
            return "False", [wiki] if wiki else []

        # Wikidata failed → fallback to Wikipedia
        wiki = query_wikipedia_summary(f"Capital of {obj}")
        if wiki:
            if subject.lower() in wiki["text"].lower():
                return "True", [wiki]
            else:
                return "False", [wiki]

        return "Unverifiable", []

    # ---------------- COUNT ----------------
    if relation == "count":
        query = f"{claim['subject']} {claim['object']}"
        wiki = query_wikipedia_summary(query)

        if wiki:
            if str(claim["value"]) in wiki["text"]:
                return "True", [wiki]
            else:
                return "False", [wiki]

        return "Unverifiable", []

    # ---------------- FALLBACK ----------------
    query = f"{claim['subject']} {claim['object']}"
    wiki = query_wikipedia_summary(query)

    if wiki:
        return "Partially true", [wiki]

    return "Unverifiable", []
