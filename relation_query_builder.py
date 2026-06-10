def build_relation_query(claim):

    relation = claim["relation"]

    subject = claim["subject"]

    # Wikipedia page names
    if subject.lower() == "einstein":
        return "Albert Einstein"

    if subject.lower() == "hitler":
        return "Adolf Hitler"

    return subject