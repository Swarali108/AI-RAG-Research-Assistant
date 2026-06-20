"""Retrieval and ranking logic for the RAG pipeline.

This module is intentionally dependency-light and side-effect free so it can be
unit-tested offline without an API key. It provides:

- ``tokenize``           : shared tokenizer (lowercase, stopword-filtered)
- ``bm25_scores``        : Okapi BM25 lexical relevance
- ``cosine_similarity``  : semantic similarity for embedding vectors
- ``hybrid_rank``        : combine lexical + (optional) semantic scores and
                           return the top-k chunks, ranked.

The actual Gemini embedding calls and the FastAPI app live in ``app.py``; this
module only does the math, which keeps it deterministic and testable.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens longer than 2 chars, minus stop words."""
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def bm25_scores(
    query: str,
    corpus_tokens: Sequence[Sequence[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Okapi BM25 score of ``query`` against each pre-tokenized document.

    BM25 beats raw term-count cosine because it dampens very frequent terms
    (saturation via ``k1``) and normalizes for document length (via ``b``).
    """
    query_terms = [t for t in tokenize(query) if t]
    n_docs = len(corpus_tokens)

    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    doc_lengths = [len(doc) for doc in corpus_tokens]
    avg_len = sum(doc_lengths) / n_docs if n_docs else 0.0

    # Document frequency per query term.
    doc_freq: Dict[str, int] = {}
    for term in set(query_terms):
        doc_freq[term] = sum(1 for doc in corpus_tokens if term in doc)

    # Smoothed IDF (always positive).
    idf = {
        term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }

    scores: List[float] = []
    for doc, length in zip(corpus_tokens, doc_lengths):
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * (length / avg_len if avg_len else 0))
            score += idf[term] * (tf * (k1 + 1)) / denom
        scores.append(score)

    return scores


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def _min_max_normalize(values: List[float]) -> List[float]:
    """Scale values to [0, 1]. Returns zeros if all values are equal."""
    if not values:
        return []

    lo, hi = min(values), max(values)
    span = hi - lo

    if span <= 0:
        return [0.0] * len(values)

    return [(v - lo) / span for v in values]


def hybrid_rank(
    question: str,
    chunks: List[Dict[str, Any]],
    query_embedding: Optional[Sequence[float]] = None,
    chunk_embeddings: Optional[Sequence[Sequence[float]]] = None,
    top_k: int = 4,
    semantic_weight: float = 0.6,
) -> List[Dict[str, Any]]:
    """Rank ``chunks`` by a blend of BM25 (lexical) and cosine (semantic) scores.

    Each chunk dict must contain a ``"text"`` key. The returned list contains
    shallow copies of the selected chunks, each enriched with:

    - ``score``          : combined relevance in [0, 1]
    - ``lexical_score``  : raw BM25 score
    - ``semantic_score`` : raw cosine score (0.0 when no embeddings supplied)
    - ``matched_terms``  : query terms also present in the chunk

    If ``query_embedding``/``chunk_embeddings`` are omitted (or empty), ranking
    falls back to BM25 only, so the function works fully offline.
    """
    if not chunks:
        return []

    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    lexical = bm25_scores(question, corpus_tokens)

    has_semantic = bool(
        query_embedding
        and chunk_embeddings
        and len(chunk_embeddings) == len(chunks)
    )
    if has_semantic:
        semantic = [cosine_similarity(query_embedding, vec) for vec in chunk_embeddings]
    else:
        semantic = [0.0] * len(chunks)

    norm_lex = _min_max_normalize(lexical)
    norm_sem = _min_max_normalize(semantic)

    weight = semantic_weight if has_semantic else 0.0
    combined = [
        weight * s + (1 - weight) * l
        for s, l in zip(norm_sem, norm_lex)
    ]

    question_terms = set(tokenize(question))

    ranked = []
    for idx, chunk in enumerate(chunks):
        chunk_terms = set(corpus_tokens[idx])
        ranked.append(
            {
                **chunk,
                "score": combined[idx],
                "lexical_score": lexical[idx],
                "semantic_score": semantic[idx],
                "matched_terms": sorted(question_terms & chunk_terms),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)

    selected = [c for c in ranked[:top_k] if c["score"] > 0]
    if not selected:
        selected = ranked[: min(top_k, len(ranked))]

    return selected
