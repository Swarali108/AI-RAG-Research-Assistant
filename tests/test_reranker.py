"""Tests for the feature-based reranking layer (#3). Offline."""

from src.reranker import _phrase_match, _proximity, rerank
from src.retrieval import tokenize


def _chunk(text, score=0.5, section=""):
    return {"text": text, "score": score, "section": section,
            "source": "doc.pdf", "page": 1, "chunk_id": "c"}


def test_rerank_promotes_exact_phrase_over_higher_base_score():
    candidates = [
        _chunk("random words about other unrelated topics here", score=0.9),
        _chunk("the transformer architecture uses self attention", score=0.6),
    ]
    out = rerank("transformer architecture", candidates, top_k=2)
    # the exact-phrase chunk should win despite a lower base score
    assert out[0]["text"].startswith("the transformer architecture")
    assert "rerank_score" in out[0] and "rerank_features" in out[0]


def test_rerank_rewards_heading_match():
    candidates = [
        _chunk("body text mentioning budget once", score=0.5, section="Introduction"),
        _chunk("body text mentioning budget once", score=0.5, section="Budget Planning"),
    ]
    out = rerank("budget", candidates, top_k=2)
    assert out[0]["section"] == "Budget Planning"


def test_rerank_respects_top_k_and_empty():
    candidates = [_chunk(f"chunk number {i} retrieval", score=0.1 * i) for i in range(6)]
    assert len(rerank("retrieval", candidates, top_k=3)) == 3
    assert rerank("anything", []) == []


def test_phrase_match_levels():
    assert _phrase_match("self attention", "uses self attention here") == 1.0
    assert _phrase_match("self attention layer", "self attention is used") == 0.5
    assert _phrase_match("banana", "no overlap at all") == 0.0


def test_proximity_higher_when_terms_are_adjacent():
    near = _proximity(tokenize("alpha beta"), tokenize("alpha beta gamma delta epsilon"))
    far = _proximity(tokenize("alpha beta"), tokenize("alpha one two three four five beta"))
    assert near > far
    assert _proximity(tokenize("alpha"), tokenize("alpha beta")) == 0.0  # single term
