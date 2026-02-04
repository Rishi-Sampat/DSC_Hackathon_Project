from spellchecker import SpellChecker
import re

spell = SpellChecker()

# Domain-specific words we NEVER want to auto-correct
PROTECTED_WORDS = {
    "india", "delhi", "mumbai", "cheetah", "kangaroo",
    "hitler", "einstein", "musk", "ambani",
    "wikidata", "wikipedia"
}

def normalize_text(text: str) -> str:
    """
    Fixes minor spelling mistakes and typos.
    Does NOT aggressively rewrite the sentence.
    """

    text = text.strip()

    words = re.findall(r"\b\w+\b", text.lower())
    corrected_words = []

    for word in words:
        if word in PROTECTED_WORDS:
            corrected_words.append(word)
            continue

        # If word is known, keep it
        if word in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word))

    # Reconstruct sentence
    corrected_text = " ".join(corrected_words)

    return corrected_text
