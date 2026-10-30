"""Tests for document insights (topics + suggested questions) and the
/api/sources/analyze endpoint. LLM and embedding calls are stubbed: offline."""

import io

from fastapi.testclient import TestClient

import src.app as app_module
from src.app import _as_list
from src.insights import extract_keyphrases, fallback_questions, parse_insights

LLM_PRIMER = [
    "Deep learning uses neural networks. Machine learning learns from data. Transformers power "
    "every LLM. A vector database supports similarity search over embeddings.",
    "Retrieval-augmented generation (RAG) adds documents to the prompt. Vector databases store "
    "embeddings for similarity search. RAG reduces hallucinations. Its knowledge cutoff matters.",
    "Fine-tuning changes model weights; RAG changes the prompt. Knowledge cutoff limits what "
    "LLMs know. Does the model read words? No, it reads tokens. Next, part two.",
]


def test_keyphrases_are_meaningful_not_stopwords():
    phrases = extract_keyphrases(LLM_PRIMER)
    lowered = [p.lower() for p in phrases]

    for junk in ("its", "does", "next", "part", "how", "you"):
        assert junk not in lowered
    assert "RAG" in phrases
    assert any(p.lower() in ("vector databases", "vector database") for p in phrases)
    assert "Similarity Search" in phrases


def test_keyphrases_split_on_verbs():
    phrases = extract_keyphrases(LLM_PRIMER)
    # "RAG reduces hallucinations" must not survive as one phrase.
    assert not any("reduces" in p.lower() or "changes" in p.lower() for p in phrases)


def test_parse_insights_tolerates_fences_and_prose():
    raw = 'Sure!\n```json\n{"topics": ["RAG", "Embeddings"], "questions": ["What is RAG?"]}\n```'
    assert parse_insights(raw) == {"topics": ["RAG", "Embeddings"], "questions": ["What is RAG?"]}
    assert parse_insights("no json here") is None
    assert parse_insights('{"topics": [], "questions": []}') is None


def test_fallback_questions_always_five():
    assert len(fallback_questions(["RAG", "Embeddings", "Chunking"])) == 5
    assert len(fallback_questions([])) == 5


def test_as_list_accepts_single_value_or_list():
    assert _as_list(None) == []
    assert _as_list(" https://a.com ") == ["https://a.com"]
    assert _as_list(["https://a.com", "", "https://b.com"]) == ["https://a.com", "https://b.com"]


def _stub_models(monkeypatch, llm_output):
    monkeypatch.setattr(app_module, "embed_texts", lambda texts, task_type=None: None)
    monkeypatch.setattr(app_module, "generate_answer", lambda prompt, **kwargs: llm_output)


def test_analyze_reports_each_source_and_insights(monkeypatch):
    _stub_models(monkeypatch, '{"topics": ["RAG", "Vector Databases"], "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]}')
    client = TestClient(app_module.app)
    files = [
        ("files", ("notes.txt", io.BytesIO(" ".join(LLM_PRIMER).encode() * 3), "text/plain")),
        ("files", ("scan.png", io.BytesIO(b"not an image"), "image/png")),
        ("files", ("deck.pptx", io.BytesIO(b"x"), "application/octet-stream")),
    ]

    data = client.post("/api/sources/analyze", files=files).json()

    by_name = {item["input"]: item for item in data["sources"]}
    assert by_name["notes.txt"]["status"] == "ready"
    assert by_name["notes.txt"]["pages"] == 1 and by_name["notes.txt"]["chunks"] >= 1
    assert by_name["scan.png"]["status"] == "failed"
    assert by_name["deck.pptx"]["status"] == "failed"
    assert data["topics"] == ["RAG", "Vector Databases"]
    assert len(data["questions"]) == 5
    assert data["insights_source"] == "llm"


def test_analyze_falls_back_to_keyphrases_when_llm_fails(monkeypatch):
    _stub_models(monkeypatch, "sorry, no JSON")
    client = TestClient(app_module.app)
    files = [("files", ("primer.txt", io.BytesIO(" ".join(LLM_PRIMER).encode() * 2), "text/plain"))]

    data = client.post("/api/sources/analyze", files=files).json()

    assert data["insights_source"] == "keyphrases"
    assert "RAG" in data["topics"]
    assert len(data["questions"]) == 5


def test_analyze_all_failed_still_returns_200():
    client = TestClient(app_module.app)
    files = [("files", ("broken.png", io.BytesIO(b"nope"), "image/png"))]

    response = client.post("/api/sources/analyze", files=files)

    assert response.status_code == 200
    assert response.json()["sources"][0]["status"] == "failed"
    assert response.json()["topics"] == []
