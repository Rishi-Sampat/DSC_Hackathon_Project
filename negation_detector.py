NEGATION_WORDS = [
    "not",
    "never",
    "no",
    "did not",
    "does not",
    "is not",
    "was not",
    "cannot",
    "can't"
]


def contains_negation(text):

    text = text.lower()

    for word in NEGATION_WORDS:

        if word in text:
            return True

    return False