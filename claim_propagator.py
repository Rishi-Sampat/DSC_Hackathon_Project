def propagate_subject(claims):

    if not claims:
        return claims

    first_claim = claims[0]

    words = first_claim.split()

    if len(words) < 3:
        return claims

    subject = words[0]

    relation_prefix = ""

    if " died in " in first_claim.lower():
        relation_prefix = f"{subject} died in "

    elif " was born in " in first_claim.lower():
        relation_prefix = f"{subject} was born in "

    elif " was " in first_claim.lower():
        relation_prefix = f"{subject} was "

    elif " is " in first_claim.lower():
        relation_prefix = f"{subject} is "

    propagated = [first_claim]

    for claim in claims[1:]:

        claim = claim.strip()

        if subject.lower() not in claim.lower():

            claim = relation_prefix + claim

        propagated.append(claim)

    return propagated