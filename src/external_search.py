try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def search_web(query, max_results=3):
    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)

            for item in search_results:
                results.append({
                    "title": item.get("title", "Untitled"),
                    "url": item.get("href") or item.get("url", ""),
                    "snippet": item.get("body") or item.get("snippet", ""),
                })

    except Exception:
        return []

    return results
