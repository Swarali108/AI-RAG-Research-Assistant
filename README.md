# AI RAG Research Assistant

A source-grounded AI research assistant. Upload PDFs, ask questions, and get answers with citations, retrieval scores, and a live "RAG Inspector" that shows exactly how each answer was produced.

## Live Demo

_Add your deployment URL here (e.g. your Vercel app link)._

## Project Overview

This project implements a complete Retrieval Augmented Generation (RAG) pipeline as a single self-contained [FastAPI](https://fastapi.tiangolo.com/) service that is safe to deploy on serverless platforms like Vercel.

Users upload PDF documents. The system extracts text, chunks it, retrieves the most relevant chunks for a question using **hybrid retrieval** (semantic embeddings + BM25 lexical scoring), injects them into a grounded prompt, and streams an answer from an LLM — with citations back to the source file, page, and chunk. All model calls (embeddings + generation) go through [OpenRouter](https://openrouter.ai/), so the provider/model is a one-line config change.

Unlike a generic chatbot, this assistant answers from your uploaded documents and is transparent about how: it shows retrieval scores, matched terms, the exact prompt sent to the model, and a warning when an answer may not be supported by the documents.

## Key Features

- PDF upload and page-wise text extraction (`pypdf`)
- **Heading-aware chunking** that splits on section boundaries (markdown/numbered/ALL-CAPS headings) instead of blind fixed windows
- **Hybrid retrieval**: embeddings (`openai/text-embedding-3-small` via OpenRouter) for semantic similarity, combined with BM25 lexical scoring, with automatic fallback to BM25-only when embeddings are unavailable
- Per-document caching keyed by file content hash, so a PDF is parsed and embedded once per session instead of on every question
- **Streamed** answer generation (Server-Sent Events) using `google/gemini-2.5-flash-lite` via OpenRouter
- **Confidence scoring** from retrieval signals, with a clear low-confidence warning when an answer may not be grounded
- **Source preview + citation highlighting**: see the exact document text used for each answer, with matched query terms highlighted
- **Six Explain-Like modes**: Research, Beginner, Interview, Professor, 30-Second Summary, and Bestie — each with its own tone, temperature, and length budget
- **Smart, document-aware follow-up suggestions** (generated from retrieval, no extra LLM cost)
- **In-app observability** (`/api/metrics`): per-request latency, token usage, estimated cost, and session fallback rate
- **RAG Inspector** UI: query terms, retrieved chunks with similarity scores, matched terms, confidence, request metrics, and the full prompt sent to the model
- Single-page UI served directly by the backend — no separate frontend build

## Production-grade capabilities

This project also demonstrates the wider AI-engineering stack — RAG, agents, evaluation, memory, and LLMOps:

| Capability | What it does | Where |
| --- | --- | --- |
| **Feature-based reranking** | Re-scores a wide hybrid candidate pool on exact-phrase, term-coverage, heading, and proximity signals (no LLM) | [`src/reranker.py`](src/reranker.py) |
| **Evaluation harness** | Hit Rate / MRR / Recall@K / Precision@K (free) + opt-in LLM-judge Faithfulness & Answer Relevance | [`src/evaluation.py`](src/evaluation.py), `GET /api/eval` |
| **Agentic query router** | Routes each question to Document RAG, free DuckDuckGo Web Search, or Conversation Memory | [`src/router.py`](src/router.py) |
| **Multi-format ingestion** | PDF · DOCX · TXT · Markdown · images (OCR) · web-page URLs · YouTube transcripts | [`src/ingestion.py`](src/ingestion.py) |
| **Conversation memory** | Sends recent turns; compresses long histories to a summary past a threshold to cut tokens | [`src/memory.py`](src/memory.py) |
| **Persistence & auth** | Supabase profiles, saved documents/chats, pgvector embeddings, persisted metrics — graceful no-op until configured | [`src/storage.py`](src/storage.py), [`supabase/schema.sql`](supabase/schema.sql) |
| **Observability** | Per-request latency, tokens, est. cost, fallback/route | `GET /api/metrics` |

> **Budget posture:** routing, reranking, confidence, follow-ups, and retrieval-metric evaluation add **zero** per-request LLM cost. The only paid LLM calls are embeddings + answer generation (and, opt-in, history compression and judge-based eval). See [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) to enable persistence.

### Key API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/research/stream` | Streamed answer (SSE) — accepts files, `url`, `youtube_url`, `use_web`, `history` |
| `POST /api/research` | Non-streamed answer with full trace |
| `GET /api/eval?k=4&judge=false` | Run the bundled evaluation set |
| `GET /api/metrics` | Session observability counters |
| `GET /api/account/status` · `/api/workspace/{documents,chats}` | Persistent workspaces (Supabase) |

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
  -> LLM via OpenRouter (streamed) -> Answer + Citations
```

Retrieval and ranking logic lives in [`src/retrieval.py`](src/retrieval.py) (pure, dependency-light, unit-tested). The FastAPI app, embedding integration, caching, and UI live in [`src/app.py`](src/app.py).

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
| `OPENROUTER_MODEL` | no | Generation model (default `google/gemini-2.5-flash-lite`) |
| `OPENROUTER_EMBEDDING_MODEL` | no | Embedding model (default `openai/text-embedding-3-small`) |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | no | Cap on answer length to control output cost (default `800`) |
| `MAX_EMBED_CHUNKS` | no | Skip semantic embeddings above this many chunks; use free BM25 instead (default `250`) |
| `ALLOWED_ORIGINS` | no | Comma-separated list of allowed CORS origins (defaults to `*`) |
| `MAX_UPLOAD_MB` | no | Per-file upload size limit in MB (default `15`) |
| `SUPABASE_URL` / `SUPABASE_KEY` | no | Enable persistent workspaces, pgvector, and persisted metrics (see [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md)) |

> **Cost note:** with the default models, embedding a document costs a fraction of a cent (and is cached), and each answer is well under a cent — a $1 OpenRouter budget covers thousands of questions. If embeddings ever fail, retrieval automatically falls back to BM25-only so the app keeps working.

## Testing

```bash
pip install -r requirements.txt
pytest
```

The retrieval logic is covered by deterministic unit tests in [`tests/`](tests/) that run fully offline (no API key required). CI runs them on every push via GitHub Actions.
