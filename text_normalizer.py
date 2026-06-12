from spellchecker import SpellChecker
import re

# -------------------------------------------------
# SPELL CHECKER
# -------------------------------------------------
spell = SpellChecker()

# -------------------------------------------------
# WORDS WE NEVER WANT TO AUTO-CORRECT
# -------------------------------------------------
PROTECTED_WORDS = {
    # Countries / Cities
    "india",
    "delhi",
    "mumbai",
    "rajkot",
    "gujarat",
    "france",
    "paris",
    "china",
    "japan",
    "london",

    # People
    "einstein",
    "hitler",
    "musk",
    "ambani",
    "gandhi",
    "tesla",
    "newton",

    # Animals
    "kangaroo",
    "cheetah",
    "elephant",
    "giraffe",

    # AI / Tech
    "chatgpt",
    "openai",
    "ollama",
    "llama",
    "gpt",
    "wikidata",
    "wikipedia",

    # Embedded / Electronics
    "stm32",
    "arduino",
    "uart",
    "spi",
    "i2c",
    "gpio",
    "pwm",
    "adc",
    "dac",
    "rs232",
    "rs485",
    "8052",

    # Organizations
    "ieee",
    "nasa",
    "isro",

    # Automotive
    "bmw",
    "mahindra",
    "toyota",
    "audi",
    "mercedes",

    "k2",
    "everest",
    "mars",
    "earth",
    "jupiter",
    "pakistan",
    "china",
    "mount"
}

# -------------------------------------------------
# COMMON TYPO MAP
# -------------------------------------------------
COMMON_TYPOS = {
    "captial": "capital",
    "capitol": "capital",
    "delli": "delhi",
    "delhii": "delhi",
    "kangroo": "kangaroo",
    "cheta": "cheetah",
    "wemen": "women",
    "wiman": "women",
    "hte": "the",
    "teh": "the",
    "recieve": "receive",
    "goverment": "government",
    "enviroment": "environment",
    "definately": "definitely",
    "seperate": "separate"
}

# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------
def normalize_text(text: str) -> str:
    """
    Corrects minor spelling mistakes and typos.
    Preserves technical terms and proper nouns.
    """
    # Safety checks
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    words = re.findall(r"\b[\w'-]+\b", text.lower())

    corrected_words = []

    for word in words:

        # Protected words
        if word in PROTECTED_WORDS:
            corrected_words.append(word)
            continue

        # Known typo mappings
        if word in COMMON_TYPOS:
            corrected_words.append(COMMON_TYPOS[word])
            continue

        # Word already known
        if word in spell:
            corrected_words.append(word)
            continue

        # Protect technical/alphanumeric terms
        if any(char.isdigit() for char in word):
            corrected_words.append(word)
            continue

        # Spell correction
        suggestion = spell.correction(word)

        # spell.correction() may return None
        if suggestion is None:
            corrected_words.append(word)
        else:
            corrected_words.append(str(suggestion))

    return " ".join(corrected_words)


# -------------------------------------------------
# TESTING
# -------------------------------------------------
if __name__ == "__main__":

    tests = [
        "Delli is the captial of India",
        "kangroo has pouch",
        "wemen are bad drivers",
        "STM32 uses UART",
        "Hitler was british",
        "Mount Everest is taller than K2",
        "RS485 is faster than RS232"
    ]

    for t in tests:
        print("Original :", t)
        print("Corrected:", normalize_text(t))
        print()