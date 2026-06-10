def check_contradiction(statement: str, evidence_text: str) -> bool:
    """
    Conservative contradiction detection.

    Only return True when evidence explicitly
    contradicts the claim.
    """

    statement = statement.lower()
    evidence_text = evidence_text.lower()

    # Capital contradiction
    if "capital" in statement and "not the capital" in evidence_text:
        return True

    # Richest vs poorest
    if "poorest" in statement and "wealthiest" in evidence_text:
        return True

    if "richest" in statement and "poorest" in evidence_text:
        return True

    return False