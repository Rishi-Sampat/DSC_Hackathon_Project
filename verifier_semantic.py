from evidence_wikidata import query_wikidata_capital
from evidence_wikidata import query_wikidata_capital
from evidence_wikidata import query_wikidata_deathplace
from evidence_wikipedia import query_wikipedia_summary
from entity_resolver import entities_match
from country_aliases import country_match
from semantic_matcher import semantic_contains
from relation_query_builder import build_relation_query
from country_aliases import COUNTRY_ALIASES
from relation_query_builder import build_relation_query

def verify_structured_claim(claim):
    """
    Verifies structured claims using:
    - Wikidata
    - Wikipedia
    - Entity Resolution

    Returns:
        (truth_status, sources) 
    """

    relation = claim["relation"]
    
    # ==========================================
    # CAPITAL OF
    # ==========================================
    if relation == "capital_of":

        subject = claim["subject"]
        obj = claim["object"]

        capital = query_wikidata_capital(obj)

        if capital:

            if entities_match(subject, capital):
                wiki = query_wikipedia_summary(capital)
                return "True", [wiki] if wiki else []

            wiki = query_wikipedia_summary(capital)
            return "False", [wiki] if wiki else []

        return "Unverifiable", []

    # ==========================================
    # COUNT
    # ==========================================
    elif relation == "count":

        query = f"{claim['subject']} {claim['object']}"
        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        if str(claim["value"]) in wiki["text"]:
            return "True", [wiki]

        return "False", [wiki]

    # ==========================================
    # LOCATION
    # ==========================================
    elif relation == "located_in":

        query = build_relation_query(claim)
        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()

        if claim["object"].lower() in evidence:
            return "True", [wiki]

        return "False", [wiki]

    # ==========================================
    # BORN IN
    # ==========================================
    elif relation == "born_in":

        query = build_relation_query(claim)

        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()

        target = claim["object"].lower()

        if target in evidence:
            return "True", [wiki]

        aliases = COUNTRY_ALIASES.get(target, [])

        for alias in aliases:

            if f"{alias}-born" in evidence:
                return "True", [wiki]

            if alias in evidence:
                return "True", [wiki]

        return "False", [wiki]
# ==========================================
# DIED IN
# ==========================================
    elif relation == "died_in":

        query = build_relation_query(claim)

        death_place = query_wikidata_deathplace(
            query
        )

        if not death_place:
            return "Unverifiable", []

        # Exact entity match
        if entities_match(
            claim["object"],
            death_place
        ):
            return "True", []

    # Example:
    # Claim = Berlin
    # Wikidata = Führerbunker
    # Check if Berlin appears in Führerbunker page

        death_place_wiki = query_wikipedia_summary(
            death_place
        )

        if death_place_wiki:

            evidence = death_place_wiki["text"].lower()

            if claim["object"].lower() in evidence:
                return "True", [death_place_wiki]

        return "False", []

    # ==========================================
    # NATIONALITY
    # ==========================================

    elif relation == "nationality":

        query = build_relation_query(claim)

        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()

        if country_match(claim["object"], evidence):
            return "True", [wiki]

        return "False", [wiki]    
    # ==========================================
    # OCCUPATION
    # ==========================================
    elif relation == "occupation":

        query = build_relation_query(claim)
        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()
        occupation = claim["object"].lower()

        if occupation in evidence:
            return "True", [wiki]

        return "False", [wiki]

    # ==========================================
    # INVENTED BY
    # ==========================================
    elif relation == "invented_by":

        query = build_relation_query(claim)
        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()

        if claim["subject"].lower() in evidence:
            return "True", [wiki]

        return "Partially true", [wiki]

    # ==========================================
    # IS-A
    # ==========================================
    elif relation == "is_a":

        query = build_relation_query(claim)

        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        evidence = wiki["text"].lower()

        target = claim["object"]

        if semantic_contains(target, evidence):
            return "True", [wiki]

        return "False", [wiki]
    # ==========================================
    # FALLBACK
    # ==========================================
    else:

        query = claim["subject"]
        wiki = query_wikipedia_summary(query)

        if wiki:
            return "Partially true", [wiki]

        return "Unverifiable", []