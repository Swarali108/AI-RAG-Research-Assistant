"""Tests for the FastAPI app's offline helpers (no Gemini API calls)."""

from pathlib import Path

from src.app import (
    _document_signature,
    build_rag_prompt,
    chunk_pages,
    load_pdf_pages,
    retrieve_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = PROJECT_ROOT / "docs" / "sample.pdf"


def test_chunk_pages_produces_well_formed_chunks_with_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    pages = [{"source": "doc.pdf", "page": 1, "text": text}]
    chunks = chunk_pages(pages, chunk_words=220, overlap=45)

    assert len(chunks) > 1  # 600 words must split into multiple chunks
    for chunk in chunks:
        assert set(chunk) >= {"source", "page", "chunk_id", "text", "top_terms"}
        assert chunk["source"] == "doc.pdf"
    # chunk ids are unique
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_document_signature_is_stable_and_content_sensitive():
    a = [{"filename": "a.pdf", "bytes": b"hello world"}]
    a_again = [{"filename": "a.pdf", "bytes": b"hello world"}]
    b = [{"filename": "a.pdf", "bytes": b"different"}]

    assert _document_signature(a) == _document_signature(a_again)
    assert _document_signature(a) != _document_signature(b)


def test_document_signature_is_order_independent():
    files = [
        {"filename": "a.pdf", "bytes": b"aaa"},
        {"filename": "b.pdf", "bytes": b"bbb"},
    ]
    reversed_files = list(reversed(files))
    assert _document_signature(files) == _document_signature(reversed_files)


def test_retrieve_chunks_lexical_fallback_makes_no_api_call():
    chunks = chunk_pages(
        [{"source": "doc.pdf", "page": 1, "text": "alpha beta gamma delta epsilon zeta"}]
    )
    # chunk_embeddings=None -> pure BM25, never touches the network
    results = retrieve_chunks("alpha gamma", chunks, chunk_embeddings=None, top_k=2)
    assert results
    assert all("score" in r for r in results)


def test_build_rag_prompt_switches_tone_and_includes_citations():
    chunks = [
        {
            "source": "doc.pdf",
            "page": 3,
            "chunk_id": "doc.pdf_page_3_chunk_1",
            "score": 0.9,
            "text": "Sample grounded content.",
            "matched_terms": ["sample"],
        }
    ]
    research = build_rag_prompt("What is this?", chunks, "Research Mode")
    bestie = build_rag_prompt("What is this?", chunks, "Bestie Mode")

    assert "Research Mode" in research
    assert "Bestie Mode" in bestie
    assert "doc.pdf" in research
    assert "Page: 3" in research


def test_load_pdf_pages_reads_sample_pdf():
    if not SAMPLE_PDF.exists():
        return  # sample asset not present; skip silently
    pages = load_pdf_pages(SAMPLE_PDF.read_bytes(), "sample.pdf")
    assert pages
    assert all(page["text"] for page in pages)
    assert pages[0]["source"] == "sample.pdf"
