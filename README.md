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

> **Cost note:** with the default models, embedding a document costs a fraction of a cent (and is cached), and each answer is well under a cent — a $1 OpenRouter budget covers thousands of questions. If embeddings ever fail, retrieval automatically falls back to BM25-only so the app keeps working.

## Testing

```bash
pip install -r requirements.txt
pytest
```

The retrieval logic is covered by deterministic unit tests in [`tests/`](tests/) that run fully offline (no API key required). CI runs them on every push via GitHub Actions.
