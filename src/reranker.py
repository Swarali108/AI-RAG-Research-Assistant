"""Feature-based reranking layer (#3).

Hybrid retrieval gives a good candidate set, but the ordering can still be
improved with signals that are cheap to compute and don't need an LLM call —
keeping the project budget-safe. We pull a wider candidate pool from the hybrid
ranker, then re-score the top candidates on:

- base       : the hybrid (semantic + BM25) score, min-max normalized
- coverage   : fraction of distinct query terms present in the chunk
- phrase     : exact query phrase / bigram appears in the chunk
- heading    : query terms appear in the chunk's section heading
- proximity  : how tightly the matched query terms cluster in the text

The blended ``rerank_score`` reorders the candidates; the top-k are returned.
This module is pure and unit-tested offline.
"""

from typing import Any, Dict, List, Sequence

try:
    from src.retrieval import tokenize
except ImportError:  # when the app dir itself is on sys.path
    from retrieval import tokenize


RERANK_WEIGHTS = {
    "base": 0.40,
    "coverage": 0.20,
    "phrase": 0.20,
    "heading": 0.10,
    "proximity": 0.10,
}


def _min_max(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def _proximity(query_terms: Sequence[str], chunk_tokens: Sequence[str]) -> float:
    """1.0 when matched query terms sit right next to each other, →0 as they
    spread out. 0.0 if fewer than two distinct query terms are present."""
    wanted = set(query_terms)
    positions = [i for i, tok in enumerate(chunk_tokens) if tok in wanted]
    distinct = {chunk_tokens[i] for i in positions}
    if len(distinct) < 2:
        return 0.0
    span = max(positions) - min(positions)
    if span <= 0:
        return 1.0
    # Ideal span if the distinct terms were adjacent is (len(distinct) - 1).
    ideal = len(distinct) - 1
    return max(0.0, min(1.0, ideal / span))


def _phrase_match(question: str, chunk_text: str) -> float:
    """1.0 if the whole normalized query appears verbatim, 0.5 if any adjacent
    query bigram appears, else 0.0."""
    q_tokens = tokenize(question)
    if not q_tokens:
        return 0.0

    norm_chunk = " ".join(tokenize(chunk_text))
    norm_query = " ".join(q_tokens)
    if len(q_tokens) >= 2 and norm_query in norm_chunk:
        return 1.0

    for a, b in zip(q_tokens, q_tokens[1:]):
        if f"{a} {b}" in norm_chunk:
            return 0.5
    return 0.0


def rerank(question: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Re-score and reorder retrieval candidates, returning the top-k enriched
    with ``rerank_score`` and a ``rerank_features`` breakdown."""
    if not candidates:
        return []

    query_terms = tokenize(question)
    query_set = set(query_terms)
    base_norm = _min_max([float(c.get("score") or 0.0) for c in candidates])

    reranked = []
    for idx, chunk in enumerate(candidates):
        chunk_tokens = tokenize(chunk.get("text", ""))
        chunk_set = set(chunk_tokens)

        coverage = (len(query_set & chunk_set) / len(query_set)) if query_set else 0.0
        phrase = _phrase_match(question, chunk.get("text", ""))
        heading_tokens = set(tokenize(chunk.get("section", "")))
        heading = 1.0 if (query_set & heading_tokens) else 0.0
        proximity = _proximity(query_terms, chunk_tokens)

        features = {
            "base": round(base_norm[idx], 3),
            "coverage": round(coverage, 3),
            "phrase": phrase,
            "heading": heading,
            "proximity": round(proximity, 3),
        }
        rerank_score = sum(RERANK_WEIGHTS[k] * features[k] for k in RERANK_WEIGHTS)

        reranked.append({**chunk, "rerank_score": round(rerank_score, 4), "rerank_features": features})

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
