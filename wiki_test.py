from evidence_wikipedia import query_wikipedia_summary

queries = [
    "Hitler",
    "India",
    "New Delhi",
    "Cheetah"
]

for q in queries:

    result = query_wikipedia_summary(q)

    print("\n====================")
    print(q)

    if result:
        print("TITLE:", result["title"])
        print("TEXT:", result["text"][:150])
    else:
        print("No result")