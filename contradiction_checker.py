def check_contradiction(statement: str, evidence_text: str) -> bool:
    """
    Returns True if evidence clearly contradicts the statement.
    """

    statement = statement.lower()
    evidence_text = evidence_text.lower()

    # Wealth contradiction
    if "poorest" in statement and "wealthiest" in evidence_text:
        return True

    # Capital contradiction
    if "capital" in statement and "not the capital" in evidence_text:
        return True

    # Generic negation
    if any(neg in evidence_text for neg in ["not", "never"]) and \
       not any(neg in statement for neg in ["not", "never"]):
        return True

    return False
