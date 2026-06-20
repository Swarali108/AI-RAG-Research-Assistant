"""Unit tests for the pure retrieval/ranking logic. Runs fully offline."""

from src.retrieval import bm25_scores, cosine_similarity, hybrid_rank, tokenize


def test_tokenize_lowercases_and_filters_stopwords_and_short_tokens():
    tokens = tokenize("The Transformer Architecture is A Big Deal!")
    assert "transformer" in tokens
    assert "architecture" in tokens
    assert "big" in tokens
    # stop words and <=2 char tokens are dropped
    assert "the" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens


def test_bm25_ranks_relevant_document_highest():
    corpus = [
        tokenize("neural networks learn representations from data"),
        tokenize("the weather today is sunny and warm"),
        tokenize("a quick brown fox jumps over the lazy dog"),
    ]
    scores = bm25_scores("how do neural networks learn", corpus)
    assert len(scores) == 3
    assert scores[0] == max(scores)
    assert scores[1] == 0.0  # no overlap with the query


def test_bm25_handles_empty_query_and_corpus():
    assert bm25_scores("", [tokenize("anything")]) == [0.0]
    assert bm25_scores("query", []) == []


def test_cosine_similarity_basic_properties():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0]) == 0.0  # length mismatch
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector


def _chunk(text):
    return {"text": text, "source": "doc.pdf", "page": 1, "chunk_id": "c"}


def test_hybrid_rank_lexical_fallback_without_embeddings():
    chunks = [
        _chunk("photosynthesis converts sunlight into chemical energy in plants"),
        _chunk("the stock market closed higher on tuesday afternoon"),
    ]
    ranked = hybrid_rank("how does photosynthesis work in plants", chunks, top_k=2)
    assert ranked[0]["text"].startswith("photosynthesis")
    assert "photosynthesis" in ranked[0]["matched_terms"]
    assert ranked[0]["semantic_score"] == 0.0  # no embeddings supplied


def test_hybrid_rank_respects_top_k_and_empty_input():
    chunks = [_chunk(f"sentence number {i} about retrieval") for i in range(5)]
    assert len(hybrid_rank("retrieval", chunks, top_k=3)) == 3
    assert hybrid_rank("anything", []) == []


def test_hybrid_rank_semantic_signal_can_override_lexical():
    # Chunk A wins lexically (contains the query term); chunk C wins semantically.
    chunks = [
        _chunk("keyword keyword keyword alpha"),   # A: lexical match
        _chunk("beta gamma delta epsilon"),        # B: nothing
        _chunk("completely different wording here"),  # C: semantic match
    ]
    query_embedding = [1.0, 0.0]
    chunk_embeddings = [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]  # only C aligns with query

    ranked = hybrid_rank(
        "keyword",
        chunks,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
        semantic_weight=0.6,
    )
    # Semantic weight (0.6) outweighs lexical (0.4), so the semantically aligned
    # chunk C should rank first even though it shares no terms with the query.
    assert ranked[0]["text"] == "completely different wording here"
    assert ranked[0]["semantic_score"] == 1.0
