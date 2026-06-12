from entity_similarity import entities_similar

print(
    entities_similar(
        "Einstein",
        "Albert Einstein"
    )
)

print(
    entities_similar(
        "Hitler",
        "Adolf Hitler"
    )
)

print(
    entities_similar(
        "USA",
        "United States"
    )
)

print(
    entities_similar(
        "Newton",
        "Albert Einstein"
    )
)