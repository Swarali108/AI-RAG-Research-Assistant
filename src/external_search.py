from ddgs import DDGS


def clean_text(value, max_chars=500):
    if not value:
        return ""

    text = str(value)
    text = text.replace("\x00", " ")
    text = " ".join(text.split())

    return text[:max_chars]


def search_web(query, max_results=3):
    results = []

    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = clean_text(item.get("title", "Untitled"), max_chars=120)
                url = clean_text(item.get("href") or item.get("url", ""), max_chars=250)
                snippet = clean_text(item.get("body") or item.get("snippet", ""), max_chars=500)

                if title or snippet:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    })

    except Exception:
        return []

    return results
