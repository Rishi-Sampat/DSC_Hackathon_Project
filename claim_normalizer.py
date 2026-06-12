import re
from semantic_relation_detector import detect_relation
from entity_linker import canonicalize_entity
from negation_detector import contains_negation

def normalize_claim(text: str) -> dict:

    text = text.strip().lower()
    negated = contains_negation(text)
    
    match = re.search(
        r"(.*?) is not the capital of (.*)",
        text
    )

    if match:
        return {
            "type": "structured",
            "relation": "capital_of",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": True
        }
    # ==========================================
    # CAPITAL RELATION
    # ==========================================
    match = re.search(r"(.*?) is the capital of (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "capital_of",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }

    # ==========================================
    # COUNT RELATION
    # ==========================================
    match = re.search(r"(.*?) has (\d+) (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "count",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(3).replace("s", "").replace("ies", "y")
            ).title(),
            "value": int(match.group(2)),
            "negated": negated
        }

    # ==========================================
    # LOCATION RELATION
    # ==========================================
    match = re.search(r"(.*?) is in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "located_in",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }
    match = re.search(
        r"(.*?) was not born in (.*)",
        text
    )

    if match:
        return {
            "type": "structured",
            "relation": "born_in",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": True
        }

    # ==========================================
    # BORN IN
    # ==========================================
    match = re.search(r"(.*?) was born in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "born_in",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }
    match = re.search(
        r"(.*?) did not die in (.*)",
        text
    )

    if match:
        return {
            "type": "structured",
            "relation": "died_in",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": True
        }

    # ==========================================
    # DIED IN
    # ==========================================
    match = re.search(r"(.*?) died in (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "died_in",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }

    # ==========================================
    # INVENTED BY
    # ==========================================
    match = re.search(r"(.*?) invented (.*)", text)

    if match:
        return {
            "type": "structured",
            "relation": "invented_by",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
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
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }

    match = re.search(
        r"(.*?) was not (.*)",
        text
    )

    if match:
        return {
            "type": "structured",
            "relation": "nationality",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": True
        }
    
    # ==========================================
# COMPARISON
# ==========================================

    match = re.search(
        r"(.*?) is (not )?(taller|higher|bigger|larger) than (.*)",
        text
    )

    if match:

        relation_map = {
            "taller": "taller_than",
            "higher": "taller_than",
            "bigger": "larger_than",
            "larger": "larger_than"
        }

        return {
            "type": "structured",
            "relation": "comparison",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),

            "object": canonicalize_entity(
                match.group(4)
            ).title(),

            "comparison": relation_map[
                match.group(3)
            ],

            "negated": bool(
                match.group(2)
            )
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
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }
    
    match = re.search(
        r"(.*?) is not a[n]? (.*)",
        text
    )

    if match:
        return {
            "type": "structured",
            "relation": "is_a",
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": True
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
            "subject": canonicalize_entity(
                match.group(1)
            ).title(),
            "object": canonicalize_entity(
                match.group(2)
            ).title(),
            "value": None,
            "negated": negated
        }

    # ==========================================
    # SEMANTIC RELATION DETECTION
    # ==========================================

    relation = detect_relation(text)

    if relation:

        # Nationality patterns
        if relation == "nationality":

            match = re.search(
                r"(.*?) (?:citizen of|nationality|citizenship) (.*)",
                text
            )

            if match:

                return {
                    "type": "structured",
                    "relation": "nationality",
                    "subject": canonicalize_entity(
                        match.group(1)
                    ).title(),
                    "object": canonicalize_entity(
                        match.group(2)
                    ).title(),
                    "value": None,
                    "negated": negated
                }

        # Occupation patterns
        elif relation == "occupation":

            match = re.search(
                r"(.*?) (?:worked as|served as|occupation|profession) (.*)",
                text
            )

            if match:

                return {
                    "type": "structured",
                    "relation": "occupation",
                    "subject": canonicalize_entity(match.group(1)).title(),
                    "object": canonicalize_entity(match.group(2)).title(),
                    "value": None,
                    "negated": negated
                }

        # Location patterns
        elif relation == "located_in":

            match = re.search(
                r"(.*?) (?:located in|situated in|inside|part of) (.*)",
                text
            )

            if match:

                return {
                    "type": "structured",
                    "relation": "located_in",
                    "subject": canonicalize_entity(match.group(1)).title(),
                    "object": canonicalize_entity(match.group(2)).title(),
                    "value": None,
                    "negated": negated
                }

# ==========================================
# SEMANTIC RELATION FALLBACK
# ==========================================

    relation = detect_relation(text)

    if relation:

        words = text.split()

        if len(words) >= 3:

            return {
                "type": "structured",
                "relation": relation,
                "subject": words[0].title(),
                "object": words[-1].title(),
                "value": None,
                "negated": negated
           }

    # ==========================================
    # FALLBACK
    # ==========================================
    return {
        "type": "unstructured",
        "relation": None,
        "subject": text,
        "object": None,
        "value": None,
        "negated": negated
    }