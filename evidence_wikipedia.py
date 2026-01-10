import wikipedia

def query_wikipedia_summary(query, sentences=2):
    try:
        page = wikipedia.page(query, auto_suggest=True)
        summary = ". ".join(page.summary.split(". ")[:sentences])
        return {
            "text": summary,
            "source": "Wikipedia",
            "url": page.url
        }
    except:
        return None
