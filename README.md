# Researchly: AI RAG Research Assistant

A source-grounded AI research assistant. Upload documents, links, videos or pasted text, ask questions, and get answers with page-level citations, retrieval scores, and a live "RAG Inspector" that shows exactly how each answer was produced.

## Project Overview

This project implements a complete Retrieval Augmented Generation (RAG) pipeline as a single self-contained [FastAPI](https://fastapi.tiangolo.com/) service that is safe to deploy on serverless platforms like Vercel.

Users upload PDF documents. The system extracts text, chunks it, retrieves the most relevant chunks for a question using **hybrid retrieval** (semantic embeddings + BM25 lexical scoring), injects them into a grounded prompt, and streams an answer from Anthropic's Claude Haiku 5.5 — with citations back to the source file, page, and chunk. All model calls (embeddings + generation) go through [OpenRouter](https://openrouter.ai/), so the provider/model is a one-line config change.

Unlike a generic chatbot, this assistant answers from your uploaded documents and is transparent about how: it shows retrieval scores, matched terms, the exact prompt sent to the model, and a warning when an answer may not be supported by the documents.

## Key Features

- PDF upload and page-wise text extraction (`pypdf`)
- **Heading-aware chunking** that splits on section boundaries (markdown/numbered/ALL-CAPS headings) instead of blind fixed windows
- **Hybrid retrieval**: embeddings (`openai/text-embedding-3-small` via OpenRouter) for semantic similarity, combined with BM25 lexical scoring, with automatic fallback to BM25-only when embeddings are unavailable
- Per-document caching keyed by file content hash, so a PDF is parsed and embedded once per session instead of on every question
- **Streamed** answer generation (Server-Sent Events) using `anthropic/claude-haiku-5.5` via OpenRouter
- **Confidence scoring** from retrieval signals, with a clear low-confidence warning when an answer may not be grounded
- **Source preview + citation highlighting**: see the exact document text used for each answer, with matched query terms highlighted
- **Six Explain-Like modes**: Research, Beginner, Interview, Professor, 30-Second Summary, and Bestie — each with its own tone and temperature
- **Smart, document-aware follow-up suggestions** (generated from retrieval, no extra LLM cost)
- **In-app observability** (`/api/metrics`): per-request latency, token usage, estimated cost, and session fallback rate
- **RAG Inspector** UI: query terms, retrieved chunks with similarity scores, matched terms, confidence, request metrics, and the full prompt sent to the model
- Single-page UI served by the backend from [`src/static/`](src/static/) (vanilla JS, no build step)

### The interface

- **Three-panel workspace** (dark theme, collapsible side panels, drawers on mobile): chats on the left, the conversation in the center, knowledge and insights on the right
- **Markdown answers** (headings, lists, tables, code), sanitized with DOMPurify because answers quote document and web text
- **Clickable source chips** under every answer, e.g. `Understanding_LLMs.pdf (p. 6)`, which open the exact passage used
- **Knowledge panel**: drag-and-drop upload, Add URL, YouTube and Paste Text, with per-source status (Processing, Ready or Failed), pages and chunks
- **Key topics and suggested questions** generated from your documents: one small LLM call per document set, with a free keyphrase extractor as fallback
- **Resilient ingestion**: one unreadable file (e.g. an image when OCR isn't installed) is skipped with a clear warning, and the answer still comes from the files that worked
- **Chats**: search, rename, delete, relative times, reopen with full context; Saved Research, thumbs feedback, regenerate, export to Markdown or PDF

## Production-grade capabilities

This project also demonstrates the wider AI-engineering stack — RAG, agents, evaluation, memory, and LLMOps:

| Capability | What it does | Where |
| --- | --- | --- |
| **Feature-based reranking** | Re-scores a wide hybrid candidate pool on exact-phrase, term-coverage, heading, and proximity signals (no LLM) | [`src/reranker.py`](src/reranker.py) |
| **Evaluation harness** | Hit Rate / MRR / Recall@K / Precision@K (free) + opt-in LLM-judge Faithfulness & Answer Relevance | [`src/evaluation.py`](src/evaluation.py), `GET /api/eval` |
| **Agentic query router** | Routes each question to Document RAG, free DuckDuckGo Web Search, or Conversation Memory | [`src/router.py`](src/router.py) |
| **Multi-format ingestion** | PDF · DOCX · TXT · Markdown · images (OCR) · web-page URLs · YouTube transcripts | [`src/ingestion.py`](src/ingestion.py) |
| **Conversation memory** | Sends the full conversation verbatim on every follow-up, with no window or summarising | [`src/memory.py`](src/memory.py) |
| **Persistence** | Neon Postgres profiles, saved documents/chats, pgvector embeddings, persisted metrics — graceful no-op until configured | [`src/storage.py`](src/storage.py), [`db/schema.sql`](db/schema.sql) |
| **Observability** | Per-request latency, tokens, est. cost, fallback/route | `GET /api/metrics` |

> **Budget posture:** routing, reranking, confidence, follow-ups, and retrieval-metric evaluation add **zero** per-request LLM cost. The paid LLM calls are embeddings, answer generation, one small topics-and-questions call per document set (~$0.0003, cached), and opt-in judge-based eval. See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md) to enable persistence.

### Key API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/research/stream` | Streamed answer (SSE) — accepts files, `url`, `youtube_url` (each may repeat), `use_web`, `history` |
| `POST /api/sources/analyze` | Index sources and report per-source status, pages and chunks, plus key topics and suggested questions |
| `POST /api/research` | Non-streamed answer with full trace |
| `GET /api/eval?k=4&judge=false` | Run the bundled evaluation set |
| `GET /api/metrics` | Session observability counters |
| `GET /api/account/status` · `/api/workspace/{documents,chats}` | Persistent workspaces (Postgres) |

## Architecture

```text
PDF Upload (multipart)
  -> Text Extraction (pypdf)
  -> Chunking (word windows + overlap)
  -> Hybrid Retrieval
       - Semantic: openai/text-embedding-3-small (via OpenRouter) + cosine similarity
       - Lexical:  BM25 over tokenized chunks
       - Combined + ranked, with BM25-only fallback
  -> Grounded Prompt Construction
  -> anthropic/claude-haiku-5.5 via OpenRouter (streamed) -> Answer + Citations
```

Retrieval and ranking logic lives in [`src/retrieval.py`](src/retrieval.py) (pure, dependency-light, unit-tested). The FastAPI app, embedding integration and caching live in [`src/app.py`](src/app.py); topic and question generation in [`src/insights.py`](src/insights.py); the UI in [`src/static/`](src/static/).

## Running Locally

```bash
python -m venv venv
venv/Scripts/activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# set your OpenRouter key (or put it in a .env file as OPENROUTER_API_KEY=sk-or-...)
export OPENROUTER_API_KEY=sk-or-your_key_here

uvicorn src.app:app --reload
```

Then open http://127.0.0.1:8000.

### Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | yes | [OpenRouter](https://openrouter.ai/keys) API key used for embeddings and answer generation |
| `OPENROUTER_MODEL` | no | Generation model (default `anthropic/claude-haiku-5.5`) |
| `OPENROUTER_EMBEDDING_MODEL` | no | Embedding model (default `openai/text-embedding-3-small`) |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | no | Optional cap on answer length to control output cost (default: no cap) |
| `MAX_EMBED_CHUNKS` | no | Skip semantic embeddings above this many chunks; use free BM25 instead (default `3000`; sent in batches of 100) |
| `ALLOWED_ORIGINS` | no | Comma-separated list of allowed CORS origins (defaults to `*`) |
| `MAX_UPLOAD_MB` | no | Per-file upload size limit in MB (default `15`) |
| `DATABASE_URL` | no | Neon Postgres connection string; enables persistent workspaces, pgvector, and persisted metrics (see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)) |

> **Cost note:** with the default models, embedding a document costs a fraction of a cent (and is cached), and each answer is well under a cent — a $1 OpenRouter budget covers several hundred questions. If embeddings ever fail, retrieval automatically falls back to BM25-only so the app keeps working.

## Testing

```bash
pip install -r requirements.txt
pytest
```

The retrieval logic is covered by deterministic unit tests in [`tests/`](tests/) that run fully offline (no API key required). CI runs them on every push via GitHub Actions.
