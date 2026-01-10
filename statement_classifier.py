def classify_statement(text: str) -> str:
    text = text.lower()

    # Comparative / extreme claims
    if any(word in text for word in [
        "poorest", "richest", "best", "worst", "largest", "smallest"
    ]):
        return "COMPARATIVE"

    # Numerical claims
    if any(word in text for word in [
        "has", "number of", "total", "count"
    ]) and any(char.isdigit() for char in text):
        return "NUMERICAL"

    # Hard factual relations
    if any(word in text for word in [
        "capital", "located", "is in", "invented"
    ]):
        return "HARD_FACT"

    # Opinion / generalization
    if any(word in text for word in [
        "always", "never", "naturally", "all", "none"
    ]):
        return "OPINION"

    return "UNVERIFIABLE"
