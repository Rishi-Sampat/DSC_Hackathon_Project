from evidence_wikidata import query_wikidata_capital
from evidence_wikidata import query_wikidata_deathplace
from evidence_wikipedia import query_wikipedia_summary
from entity_resolver import entities_match
from country_aliases import country_match
from semantic_matcher import semantic_contains
from relation_query_builder import build_relation_query
from country_aliases import COUNTRY_ALIASES
from entity_similarity import entities_similar
from comparison_facts import COMPARISON_FACTS
from numeric_facts import NUMERIC_FACTS
from temporal_facts import TEMPORAL_FACTS

def apply_negation(truth_status, claim):

    if not claim.get("negated", False):
        return truth_status

    if truth_status == "True":
        return "False"

    if truth_status == "False":
        return "True"

    return truth_status


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

            wiki = query_wikipedia_summary(capital)

            if (
                entities_match(subject, capital)
                or
                entities_similar(subject, capital)
            ):

                result = apply_negation(
                    "True",
                    claim
                )

                return result, [wiki] if wiki else []

            result = apply_negation(
                "False",
                claim
            )

            return result, [wiki] if wiki else []

        return "Unverifiable", []

        # ==========================================
    # COUNT
    # ==========================================
    elif relation == "count":

        subject = claim["subject"].lower()
        obj = claim["object"].lower()
        value = claim["value"]

        # ------------------------------------------
        # LOCAL NUMERIC FACT DATABASE
        # ------------------------------------------

        if (
            subject in NUMERIC_FACTS
            and
            obj in NUMERIC_FACTS[subject]
        ):

            actual_value = (
                NUMERIC_FACTS[subject][obj]
            )

            result = (
                "True"
                if actual_value == value
                else "False"
            )

            result = apply_negation(
                result,
                claim
            )

            return result, []

        # ------------------------------------------
        # WIKIPEDIA FALLBACK
        # ------------------------------------------

        query = f"{claim['subject']} {claim['object']}"

        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        if str(value) in wiki["text"]:

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

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

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

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

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        aliases = COUNTRY_ALIASES.get(target, [])

        for alias in aliases:

            if f"{alias}-born" in evidence:

                result = apply_negation(
                    "True",
                    claim
                )

                return result, [wiki]

            if alias in evidence:

                result = apply_negation(
                    "True",
                    claim
                )

                return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

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

        if entities_match(
            claim["object"],
            death_place
        ):

            result = apply_negation(
                "True",
                claim
            )

            return result, []

        death_place_wiki = query_wikipedia_summary(
            death_place
        )

        if death_place_wiki:

            evidence = death_place_wiki["text"].lower()

            if claim["object"].lower() in evidence:

                result = apply_negation(
                    "True",
                    claim
                )

                return result, [death_place_wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, []

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

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

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

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

    # ==========================================
    # INVENTED BY
    # ==========================================
    elif relation == "invented_by":

        query = build_relation_query(claim)
        wiki = query_wikipedia_summary(query)

        if not wiki:
            return "Unverifiable", []

        if entities_similar(
            claim["subject"],
            wiki["title"]
        ):

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]
        # ==========================================
    # TEMPORAL FACTS
    # ==========================================
    elif relation in [
        "birth_year",
        "death_year",
        "independence_year",
        "end_year"
    ]:

        subject = claim["subject"].lower()

        if subject not in TEMPORAL_FACTS:
            return "Unverifiable", []

        relation_key = relation

        if relation_key not in TEMPORAL_FACTS[subject]:
            return "Unverifiable", []

        actual_year = (
            TEMPORAL_FACTS[subject][relation_key]
        )

        claimed_year = claim["value"]

        result = (
            "True"
            if actual_year == claimed_year
            else "False"
        )

        result = apply_negation(
            result,
            claim
        )

        return result, []
    
    # ==========================================
# COMPARISON
# ==========================================
    elif relation == "comparison":

        subject = claim["subject"].lower()
        obj = claim["object"].lower()

        if (
            subject not in COMPARISON_FACTS
            or
            obj not in COMPARISON_FACTS
        ):
            return "Unverifiable", []

        comparison_type = claim["comparison"]

        if comparison_type == "taller_than":

            s = COMPARISON_FACTS[subject].get("height")
            o = COMPARISON_FACTS[obj].get("height")

            if s is None or o is None:
                return "Unverifiable", []

            result = "True" if s > o else "False"

            result = apply_negation(
                result,
                claim
            )

            return result, []

        elif comparison_type == "larger_than":

            s = (
                COMPARISON_FACTS[subject].get("area")
                or
                COMPARISON_FACTS[subject].get("diameter")
            )

            o = (
                COMPARISON_FACTS[obj].get("area")
                or
                COMPARISON_FACTS[obj].get("diameter")
            )

            if s is None or o is None:
                return "Unverifiable", []

            result = "True" if s > o else "False"

            result = apply_negation(
                result,
                claim
            )

            return result, []

        return "Unverifiable", []

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

            result = apply_negation(
                "True",
                claim
            )

            return result, [wiki]

        result = apply_negation(
            "False",
            claim
        )

        return result, [wiki]

    # ==========================================
    # FALLBACK
    # ==========================================
    else:

        query = claim["subject"]
        wiki = query_wikipedia_summary(query)

        if wiki:
            return "Partially true", [wiki]

        return "Unverifiable", []