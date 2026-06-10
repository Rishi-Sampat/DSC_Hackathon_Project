import requests

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def query_wikipedia_summary(query):

    try:

        url = WIKI_API + query.replace(" ", "_")

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                "RAV-Hallucination-Detector/1.0"
            }
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return {
            "title": data.get("title", query),
            "text": data.get("extract", ""),
            "source": "Wikipedia",
            "url": data.get("content_urls", {})
                      .get("desktop", {})
                      .get("page", "")
        }

    except Exception as e:

        print("WIKI API ERROR:", e)

        return None