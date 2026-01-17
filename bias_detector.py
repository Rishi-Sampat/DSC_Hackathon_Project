def rule_based_bias_check(text: str):
    text = text.lower()

    bias_patterns = {
        "gender": ["women are", "men are", "girls are", "boys are"],
        "social": ["poor people", "rich people", "indian people"],
        "ethical": ["disabled people", "old people"],
        "racial": ["black people", "white people"]
    }

    for bias_type, phrases in bias_patterns.items():
        for p in phrases:
            if p in text:
                return True, bias_type

    return False, "none"
