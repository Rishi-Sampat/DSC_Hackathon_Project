def entities_similar(entity1, entity2):

    if not entity1 or not entity2:
        return False

    e1 = entity1.lower().strip()
    e2 = entity2.lower().strip()

    # Exact match
    if e1 == e2:
        return True

    # Containment
    if e1 in e2:
        return True

    if e2 in e1:
        return True

    # Token overlap
    words1 = set(e1.split())
    words2 = set(e2.split())

    overlap = words1.intersection(words2)

    if len(overlap) > 0:
        return True

    return False