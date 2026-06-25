"""Agentic query router (#6).

Routes each question to the right strategy with cheap heuristics (no LLM call,
budget-safe):

- "web"    : current-events / freshness signals, or the user forced web search
- "memory" : follow-up phrasing ("it", "that", "you said") when chat history exists
- "rag"    : default — answer from the uploaded documents

Web search uses DuckDuckGo via the free ``ddgs`` package, degrading to an empty
result list when the package or network is unavailable.
"""

import re
from typing import Any, Dict, List


CURRENT_SIGNALS = {
    "latest", "today", "current", "currently", "recent", "recently", "news", "now",
    "2024", "2025", "2026", "price", "stock", "weather", "update", "updates", "live", "trending",
}
MEMORY_SIGNALS = {
    "it", "this", "that", "they", "them", "those", "these", "same", "above",
    "previous", "previously", "earlier", "former", "aforementioned",
}


def route_question(question: str, has_history: bool = False, use_web: bool = False) -> str:
    """Return one of: 'web', 'memory', 'rag'."""
    tokens = set(re.findall(r"[a-z0-9']+", (question or "").lower()))

    if use_web or (tokens & CURRENT_SIGNALS):
        return "web"
    if has_history and (tokens & MEMORY_SIGNALS):
        return "memory"
    return "rag"


def _clean(value: Any, max_chars: int = 600) -> str:
    text = str(value or "").replace("\x00", " ")
    return " ".join(text.split())[:max_chars]


def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Free DuckDuckGo web search. Returns [] if the dependency/network is absent."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return []

    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = _clean(item.get("title", "Untitled"), 160)
                url = _clean(item.get("href") or item.get("url", ""), 300)
                snippet = _clean(item.get("body") or item.get("snippet", ""), 600)
                if title or snippet:
                    results.append({"title": title, "url": url, "snippet": snippet})
    except Exception:
        return []

    return results
