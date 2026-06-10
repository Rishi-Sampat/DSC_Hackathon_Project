def semantic_contains(target, evidence):

    target = target.lower().strip()
    evidence = evidence.lower()

    # Exact match
    if target in evidence:
        return True

    # Simple plural handling
    if target + "s" in evidence:
        return True

    if target.endswith("y"):
        if target[:-1] + "ies" in evidence:
            return True

    return False