"""Tests for Phase 0 features: modes, confidence, heading-aware chunking,
smart follow-ups. All offline (no API calls)."""

from src.app import (
    MODES,
    chunk_pages,
    compute_confidence,
    follow_up_questions,
    resolve_mode,
)


def test_resolve_mode_maps_all_six_modes():
    assert resolve_mode("Research Mode") == "research"
    assert resolve_mode("Bestie Mode") == "bestie"
    assert resolve_mode("Beginner Mode") == "beginner"
    assert resolve_mode("Interview Mode") == "interview"
    assert resolve_mode("Professor Mode") == "professor"
    assert resolve_mode("30-Second Summary") == "summary"
    assert resolve_mode("something unknown") == "research"  # safe default
    assert set(MODES) == {"research", "bestie", "beginner", "interview", "professor", "summary"}


def test_confidence_low_when_no_chunks():
    result = compute_confidence([], "any question", semantic_active=True)
    assert result["score"] == 0.0
    assert result["label"] == "low"
    assert result["warning"]


def test_confidence_high_with_strong_signals():
    chunks = [
        {"semantic_score": 0.7, "matched_terms": ["photosynthesis", "plants"]},
        {"semantic_score": 0.6, "matched_terms": ["plants"]},
    ]
    result = compute_confidence(chunks, "how does photosynthesis work in plants", semantic_active=True)
    assert result["label"] == "high"
    assert result["score"] >= 0.6
    assert result["warning"] is None


def test_confidence_lexical_only_when_semantic_inactive():
    chunks = [{"semantic_score": 0.0, "matched_terms": ["alpha", "beta"]}]
    result = compute_confidence(chunks, "alpha beta", semantic_active=False)
    # both query terms covered -> full lexical signal
    assert result["score"] == 1.0
    assert result["label"] == "high"


def test_heading_aware_chunking_tags_sections():
    text = "INTRODUCTION\n" + " ".join(f"intro{i}" for i in range(40))
    text += "\nMETHODS\n" + " ".join(f"method{i}" for i in range(40))
    pages = [{"source": "doc.pdf", "page": 1, "text": text}]
    chunks = chunk_pages(pages, chunk_words=60, overlap=10)

    sections = {c["section"] for c in chunks}
    assert "INTRODUCTION" in sections
    assert "METHODS" in sections
    # the heading is prepended into the chunk text for better context
    assert any(c["text"].startswith("INTRODUCTION") for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_follow_ups_are_mode_aware_and_document_aware():
    chunks = [{"top_terms": ["transformer", "attention"]}]
    ups = follow_up_questions("Interview Mode", chunks)
    assert len(ups) <= 4
    # first suggestion is woven from the top document term
    assert "transformer" in ups[0].lower()


def test_follow_ups_without_chunks_fall_back_to_mode_templates():
    ups = follow_up_questions("Bestie Mode", None)
    assert ups
    assert len(ups) <= 4
