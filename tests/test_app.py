"""Tests for the FastAPI app's offline helpers (no Gemini API calls)."""

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

import src.app as app_module
from src.app import (
    _document_signature,
    build_document_index,
    build_rag_prompt,
    chunk_pages,
    embed_texts,
    gather_sources,
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


class _FakeEmbeddings:
    """Stands in for client.embeddings: records batch sizes, fails on request."""

    def __init__(self, fail_on_call=None):
        self.batch_sizes = []
        self.fail_on_call = fail_on_call

    def create(self, model, input):
        self.batch_sizes.append(len(input))
        if self.fail_on_call == len(self.batch_sizes):
            raise RuntimeError("embedding API down")
        # Returned out of order on purpose; each vector encodes its text.
        data = [
            SimpleNamespace(index=i, embedding=[float(text.split("-")[1])])
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=None)


def _use_fake_client(monkeypatch, fake):
    monkeypatch.setattr(
        app_module, "get_openrouter_client", lambda: SimpleNamespace(embeddings=fake)
    )


def test_embed_texts_sends_batches_of_100_and_keeps_order(monkeypatch):
    fake = _FakeEmbeddings()
    _use_fake_client(monkeypatch, fake)

    vectors = embed_texts([f"text-{i}" for i in range(250)])

    assert fake.batch_sizes == [100, 100, 50]
    assert vectors == [[float(i)] for i in range(250)]


def test_embed_texts_returns_none_if_any_batch_fails(monkeypatch):
    fake = _FakeEmbeddings(fail_on_call=2)
    _use_fake_client(monkeypatch, fake)

    assert embed_texts([f"text-{i}" for i in range(250)]) is None


def test_document_index_falls_back_to_bm25_when_embeddings_fail(monkeypatch):
    _use_fake_client(monkeypatch, _FakeEmbeddings(fail_on_call=1))
    pages = [{"source": "notes.txt", "page": 1, "text": "alpha beta gamma delta " * 80}]

    index = build_document_index([{"key": "bm25-fallback-test", "pages": pages}])

    assert index["semantic"] is False
    assert index["chunk_embeddings"] is None
    assert retrieve_chunks("alpha gamma", index["chunks"], chunk_embeddings=None, top_k=2)


def test_unreadable_file_is_skipped_with_a_warning(monkeypatch):
    _use_fake_client(monkeypatch, _FakeEmbeddings(fail_on_call=1))
    good = {"filename": "notes.txt", "bytes": b"alpha beta gamma delta " * 80}
    broken = {"filename": "scan.png", "bytes": b"not really an image"}

    index = build_document_index([good, broken])

    assert {chunk["source"] for chunk in index["chunks"]} == {"notes.txt"}
    assert [w["source"] for w in index["warnings"]] == ["scan.png"]


def test_request_fails_only_when_no_source_is_readable():
    broken = {"filename": "scan.png", "bytes": b"still not an image"}

    with pytest.raises(HTTPException) as caught:
        build_document_index([broken])

    assert caught.value.status_code == 400
    assert "scan.png" in caught.value.detail


def test_gather_sources_skips_unsupported_and_oversized_files(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    uploads = [
        UploadFile(file=io.BytesIO(b"tiny"), filename="ok.txt"),
        UploadFile(file=io.BytesIO(b"x"), filename="slides.pptx"),
        UploadFile(file=io.BytesIO(b"way more than ten bytes"), filename="big.txt"),
    ]

    sources, warnings = asyncio.run(gather_sources(uploads))

    assert [s["filename"] for s in sources] == ["ok.txt"]
    assert [w["source"] for w in warnings] == ["slides.pptx", "big.txt"]
