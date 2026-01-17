def classify_statement(text: str) -> str:
    text = text.strip().lower()

    # -------------------------------------------------
    # 0. QUESTION / OPINION-SEEKING (EARLY EXIT)
    # -------------------------------------------------
    if text.endswith("?"):
        return "QUESTION"

    question_starters = [
        "what", "why", "how", "who", "when", "where", "which",
        "do you", "can you", "could you", "should we", "tell me"
    ]

    for q in question_starters:
        if text.startswith(q):
            return "QUESTION"

    opinion_requests = [
        "what do you think",
        "your opinion",
        "do you believe",
        "i think",
        "in my opinion"
    ]

    for op in opinion_requests:
        if op in text:
            return "OPINION_REQUEST"

    # -------------------------------------------------
    # 1. COMPARATIVE / EXTREME CLAIMS
    # -------------------------------------------------
    if any(word in text for word in [
        "poorest", "richest", "best", "worst", "largest", "smallest",
        "fastest", "slowest", "most", "least"
    ]):
        return "COMPARATIVE"

    # -------------------------------------------------
    # 2. NUMERICAL CLAIMS
    # -------------------------------------------------
    if any(word in text for word in [
        "has", "number of", "total", "count"
    ]) and any(char.isdigit() for char in text):
        return "NUMERICAL"

    # -------------------------------------------------
    # 3. HARD FACTUAL RELATIONS
    # -------------------------------------------------
    if any(word in text for word in [
        "capital", "located", "is in", "invented", "founded",
        "discovered", "born", "died"
    ]):
        return "HARD_FACT"

    # -------------------------------------------------
    # 4. GENERALIZATION / STEREOTYPE OPINIONS
    # -------------------------------------------------
    if any(word in text for word in [
        "always", "never", "naturally", "all", "none",
        "everyone", "nobody"
    ]):
        return "OPINION"

    # -------------------------------------------------
    # 5. FALLBACK
    # -------------------------------------------------
    return "UNVERIFIABLE"
