# semantic_relation_detector.py

RELATION_PATTERNS = {

    "nationality": [

        "nationality",
        "citizen of",
        "citizenship",
        "belongs to",
        "belonged to",
        "originated from",
        "from germany",
        "from france",
        "from india"
    ],

    "occupation": [

        "profession",
        "worked as",
        "served as",
        "occupation",
        "career as",
        "job as"
    ],

    "located_in": [

        "located in",
        "situated in",
        "part of",
        "inside",
        "within"
    ]
}


def detect_relation(text):

    text = text.lower()

    for relation, patterns in RELATION_PATTERNS.items():

        for pattern in patterns:

            if pattern in text:
                return relation

    return None