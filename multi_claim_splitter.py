import re

def split_claims(text):

    text = text.strip()

    connectors = []

    matches = re.finditer(
        r"\s+(and|or|but|while)\s+",
        text.lower()
    )

    for m in matches:
        connectors.append(m.group(1))

    claims = re.split(
        r"\s+(?:and|or|but|while)\s+|;",
        text
    )

    claims = [
        c.strip()
        for c in claims
        if c.strip()
    ]

    return claims, connectors