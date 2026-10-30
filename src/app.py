import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import NOT_GIVEN, OpenAI
from pydantic import BaseModel

try:
    from src.retrieval import hybrid_rank, tokenize
    from src.reranker import rerank
    from src.evaluation import (
        aggregate_metrics,
        judge_answer_relevance,
        judge_faithfulness,
        query_retrieval_metrics,
    )
except ImportError:  # when the app dir itself is on sys.path (some deploy setups)
    from retrieval import hybrid_rank, tokenize
    from reranker import rerank
    from evaluation import (
        aggregate_metrics,
        judge_answer_relevance,
        judge_faithfulness,
        query_retrieval_metrics,
    )

try:
    from src.ingestion import (
        SUPPORTED_UPLOAD_EXTENSIONS,
        extract_pages,
        extract_url,
        extract_youtube,
    )
    from src.router import route_question, web_search
    from src.memory import format_history, has_history
    from src.insights import (
        build_insights,
        build_insights_prompt,
        extract_keyphrases,
        parse_insights,
        sample_excerpts,
    )
except ImportError:
    from ingestion import (
        SUPPORTED_UPLOAD_EXTENSIONS,
        extract_pages,
        extract_url,
        extract_youtube,
    )
    from router import route_question, web_search
    from memory import format_history, has_history
    from insights import (
        build_insights,
        build_insights_prompt,
        extract_keyphrases,
        parse_insights,
        sample_excerpts,
    )

try:
    from src.storage import store
except ImportError:
    from storage import store


load_dotenv()

app = FastAPI(
    title="AI RAG Research Assistant",
    version="1.2.0",
)

# CORS origins are configurable; default stays permissive for the public demo,
# but a deployment can lock it down via ALLOWED_ORIGINS=https://your-app.vercel.app
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env in ("", "*") else [
    origin.strip() for origin in _origins_env.split(",") if origin.strip()
]
# Credentials cannot be combined with the "*" wildcard per the CORS spec.
ALLOW_CREDENTIALS = ALLOWED_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# All LLM calls go through OpenRouter (OpenAI-compatible API).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Generation: strong at grounded, well-structured answers at ~$0.001 per question.
GENERATION_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-5.5")
# Claude Haiku 5.5 thinks before answering by default on OpenRouter, and those
# hidden reasoning tokens count against max_tokens: at 800 they consumed ~740,
# leaving a ~25-word answer. Grounded RAG answers don't need it, so turn it off.
GENERATION_EXTRA_BODY = {"reasoning": {"enabled": False}}
# Embeddings: nearly free, powers hybrid semantic retrieval.
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
# Optional attribution headers shown on OpenRouter dashboards.
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/Swarali108/AI-RAG-Research-Assistant"),
    "X-Title": os.getenv("OPENROUTER_SITE_NAME", "AI RAG Research Assistant"),
}

# --- Cost guardrails (keep a small budget lasting) ---
# No answer-length cap by default: the model stops when the answer is complete.
# Set OPENROUTER_MAX_OUTPUT_TOKENS to reintroduce a ceiling if costs need reining in.
_max_output_env = os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "").strip()
GENERATION_MAX_TOKENS = int(_max_output_env) if _max_output_env else None
# Skip semantic embeddings above this many chunks (fall back to free BM25).
# 3000 chunks ≈ 1,000 pages ≈ 900k tokens ≈ $0.02, paid once per document (cached).
MAX_EMBED_CHUNKS = int(os.getenv("MAX_EMBED_CHUNKS", "3000"))
# Texts per embeddings request; embed_texts splits larger inputs into batches.
EMBED_BATCH_SIZE = 100

# --- Explain-Like modes (#13): pure prompt engineering, no extra LLM cost ---
MODES: Dict[str, Dict[str, Any]] = {
    "research": {
        "label": "Research Mode",
        "temperature": 0.2,
        "tone": (
            "Use Research Mode. Be professional, factual, concise, and recruiter-friendly. "
            "Use clear structure and headings, avoid slang, and prioritize accuracy."
        ),
        "followups": [
            "What evidence supports this answer?",
            "Can you summarize the key points?",
            "What are the limitations or caveats?",
        ],
    },
    "bestie": {
        "label": "Bestie Mode",
        "temperature": 0.65,
        "tone": (
            "Use Bestie Mode. Be sassy, fun, and lightly chaotic, using casual internet "
            "language naturally (bestie, ngl, lowkey, fr, the tea, TL;DR) without overdoing it. "
            "Stay accurate, cite sources, never invent facts — useful first, funny second."
        ),
        "followups": [
            "Spill the TL;DR.",
            "Explain this like I'm sleep-deprived.",
            "What's the actual tea here?",
        ],
    },
    "beginner": {
        "label": "Beginner Mode",
        "temperature": 0.4,
        "tone": (
            "Use Beginner Mode. Assume no prior knowledge. Use simple words and short sentences, "
            "define every piece of jargon in plain language, and use everyday analogies. Be patient "
            "and encouraging. Never sacrifice accuracy."
        ),
        "followups": [
            "Can you explain that with a simple analogy?",
            "What does this term mean in plain English?",
            "Why does this matter?",
        ],
    },
    "interview": {
        "label": "Interview Mode",
        "temperature": 0.3,
        "tone": (
            "Use Interview Mode. Optimize for interview preparation: lead with the crisp answer, then "
            "tight bullet points of the key facts worth remembering. Highlight common follow-up angles "
            "and gotchas. Be precise and structured."
        ),
        "followups": [
            "What's a likely follow-up question on this?",
            "How would I explain this in 60 seconds?",
            "What's the most common misconception here?",
        ],
    },
    "professor": {
        "label": "Professor Mode",
        "temperature": 0.3,
        "tone": (
            "Use Professor Mode. Be rigorous and in-depth with an academic tone. Define terms precisely, "
            "explain mechanisms and reasoning, note nuances, assumptions, and caveats, and structure the "
            "answer logically. Cite the source material carefully."
        ),
        "followups": [
            "Can you go deeper on the underlying mechanism?",
            "What assumptions does this rely on?",
            "How does this connect to broader theory?",
        ],
    },
    "summary": {
        "label": "30-Second Summary",
        "temperature": 0.2,
        "tone": (
            "Use 30-Second Summary Mode. Give the absolute essentials only: a one-line takeaway followed "
            "by at most 3-5 short bullet points. No preamble, no filler. Fast to read."
        ),
        "followups": [
            "Give me the full breakdown.",
            "What's the single most important point?",
            "What should I read next?",
        ],
    },
}
DEFAULT_MODE = "research"


def resolve_mode(answer_mode: Optional[str]) -> str:
    """Map a free-form answer_mode label from the UI to a MODES key."""
    a = (answer_mode or "").lower()
    if "bestie" in a or "friendly" in a:
        return "bestie"
    if "beginner" in a or "eli5" in a or "simple" in a:
        return "beginner"
    if "interview" in a:
        return "interview"
    if "professor" in a or "academic" in a:
        return "professor"
    if "30" in a or "summary" in a or "tl;dr" in a or "tldr" in a:
        return "summary"
    return DEFAULT_MODE


# --- Cost/observability constants (#10): USD per token for the default models ---
PRICE_PER_TOKEN = {
    "embedding_in": 0.02 / 1_000_000,
    "generation_in": 0.10 / 1_000_000,
    "generation_out": 0.50 / 1_000_000,
}

# In-process metrics. Resets on cold start (fine for the lite version; Phase 2
# persists these to Postgres). Surfaced via /api/metrics and the RAG Inspector.
METRICS: Dict[str, float] = {
    "requests": 0,
    "semantic_requests": 0,
    "fallback_requests": 0,
    "embedding_tokens": 0,
    "generation_prompt_tokens": 0,
    "generation_completion_tokens": 0,
    "errors": 0,
    "total_cost_usd": 0.0,
}

# Cache parsed chunks + their embeddings keyed by a hash of the uploaded bytes,
# so the same document is not re-parsed and re-embedded on every question.
_DOCUMENT_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_ENTRIES = 32


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []
    answer_mode: Optional[str] = "Research Mode"


class ChatResponse(BaseModel):
    answer: str
    follow_up_questions: List[str] = []


class ChatSave(BaseModel):
    title: str = "Untitled chat"
    turns: List[Dict[str, Any]] = []


def get_optional_user(authorization: Optional[str] = Header(None)):
    """Resolve the access token (if any) to a user; None when anonymous
    or when persistence is not configured."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return store.user_from_token(authorization.split(" ", 1)[1].strip())


def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file (or Vercel "
            "Environment Variables) as OPENROUTER_API_KEY=sk-or-..."
        )

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key.strip(),
        default_headers=OPENROUTER_HEADERS,
    )


def embed_texts(texts: List[str], task_type: Optional[str] = None) -> Optional[List[List[float]]]:
    """Embed texts via OpenRouter. Returns None on any failure so callers can
    gracefully fall back to lexical-only (BM25) retrieval. ``task_type`` is kept
    for call-site clarity but is not required by the embedding model."""
    if not texts:
        return []

    # Batched so large documents stay under per-request input limits. All or
    # nothing: if any batch fails, return None rather than a partial list that
    # would no longer line up one-to-one with the chunks.
    embeddings: List[List[float]] = []
    try:
        client = get_openrouter_client()
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            usage = getattr(response, "usage", None)
            if usage is not None:
                METRICS["embedding_tokens"] += getattr(usage, "total_tokens", 0) or 0
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                return None
            embeddings.extend(item.embedding for item in data)
        return embeddings
    except Exception:
        return None


def load_pdf_pages(file_bytes: bytes, source_name: str) -> List[Dict[str, Any]]:
    """Backward-compatible PDF loader; delegates to the multi-format ingester."""
    return extract_pages(source_name, file_bytes)


_HEADING_RE = re.compile(r"^(#{1,6}\s+\S|\d+(\.\d+)*[.)]\s+\S)")


def _is_heading(line: str) -> bool:
    """Conservative heading detector: markdown, numbered, ALL-CAPS, or a short
    line ending in a colon. Tuned to avoid flagging ordinary prose."""
    words = line.split()
    if not (1 <= len(words) <= 12):
        return False
    if _HEADING_RE.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 2 and line.upper() == line and len(words) <= 10:
        return True
    if len(words) <= 8 and line.endswith(":"):
        return True
    return False


def _split_sections(page_text: str):
    """Split a page into (heading, body) sections on detected heading lines."""
    sections = []
    current_heading = None
    buffer: List[str] = []

    for line in page_text.split("\n"):
        if _is_heading(line):
            if buffer:
                sections.append((current_heading, " ".join(buffer)))
                buffer = []
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip().rstrip(":")
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_heading, " ".join(buffer)))

    if not sections:
        sections = [(None, page_text.replace("\n", " "))]

    return sections


def chunk_pages(pages: List[Dict[str, Any]], chunk_words: int = 220, overlap: int = 45):
    """Heading-aware chunking: split each page into sections, then window within
    each section so chunks don't straddle topic boundaries. The section heading
    is prepended to the chunk text to give retrieval and the LLM better context."""
    chunks = []

    for page in pages:
        chunk_index = 1

        for heading, body in _split_sections(page["text"]):
            words = body.split()
            if not words:
                continue

            start = 0
            while start < len(words):
                end = start + chunk_words
                window = " ".join(words[start:end])

                if window:
                    text = f"{heading}\n{window}" if heading else window
                    chunks.append(
                        {
                            "source": page["source"],
                            "page": page["page"],
                            "section": heading or "",
                            "chunk_id": f"{page['source']}_page_{page['page']}_chunk_{chunk_index}",
                            "text": text,
                            "top_terms": [term for term, _ in Counter(tokenize(text)).most_common(8)],
                        }
                    )
                    chunk_index += 1

                if end >= len(words):
                    break

                start = max(0, end - overlap)

    return chunks


def _document_signature(sources: List[Dict[str, Any]]) -> str:
    """Stable cache key over the input sources (file bytes, or pre-extracted
    pages for URLs/YouTube)."""
    hasher = hashlib.sha256()
    for source in sorted(sources, key=lambda item: item.get("filename") or item.get("key", "")):
        if "bytes" in source:
            hasher.update((source.get("filename") or "").encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(source["bytes"])
        else:
            hasher.update((source.get("key") or "").encode("utf-8"))
            hasher.update(b"\0")
            for page in source.get("pages", []):
                hasher.update(page["text"].encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


class NoReadableSources(HTTPException):
    """400 raised when nothing could be extracted; keeps the per-source reasons."""

    def __init__(self, warnings: List[Dict[str, str]]):
        super().__init__(status_code=400, detail=_no_readable_sources_message(warnings))
        self.warnings = warnings


def _no_readable_sources_message(warnings: List[Dict[str, str]]) -> str:
    if not warnings:
        return "No readable text was found in the provided sources."
    skipped = "; ".join(f"{w['source']}: {w['message']}" for w in warnings)
    return f"None of your sources could be read. {skipped}"


def build_document_index(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse, chunk, and embed the input sources once, caching by content hash.
    Each source is either an uploaded file ({"filename","bytes"}) or a set of
    pre-extracted pages ({"key","pages"}) from a URL or YouTube transcript."""
    signature = _document_signature(sources)
    cached = _DOCUMENT_CACHE.get(signature)
    if cached is not None:
        return cached

    # One unreadable file must not sink the request: skip it, record why, and
    # answer from whatever did extract.
    pages = []
    warnings: List[Dict[str, str]] = []
    source_pages: Dict[str, int] = {}  # per-source page counts, for the sources panel
    for source in sources:
        if "bytes" not in source:
            source_items = source.get("pages", [])
        else:
            try:
                source_items = extract_pages(source["filename"], source["bytes"])
            except Exception as error:
                warnings.append({"source": source["filename"], "message": str(error) or "could not be read."})
                continue
            if not source_items:
                warnings.append({
                    "source": source["filename"],
                    "message": "no readable text was found (a scanned PDF needs OCR).",
                })
        for page in source_items:
            source_pages[page["source"]] = source_pages.get(page["source"], 0) + 1
        pages.extend(source_items)

    if not pages:
        raise NoReadableSources(warnings)

    chunks = chunk_pages(pages)

    # Cost guard: only embed documents up to MAX_EMBED_CHUNKS; larger ones use
    # BM25-only retrieval (free) instead of paying to embed thousands of chunks.
    if len(chunks) <= MAX_EMBED_CHUNKS:
        chunk_embeddings = embed_texts(
            [chunk["text"] for chunk in chunks], task_type="RETRIEVAL_DOCUMENT"
        )
    else:
        chunk_embeddings = None

    index = {
        "pages": len(pages),
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "semantic": chunk_embeddings is not None,
        "warnings": warnings,
        "source_pages": source_pages,
    }

    # Simple FIFO bound so a long-lived process does not grow without limit.
    if len(_DOCUMENT_CACHE) >= _CACHE_MAX_ENTRIES:
        _DOCUMENT_CACHE.pop(next(iter(_DOCUMENT_CACHE)))
    _DOCUMENT_CACHE[signature] = index

    return index


def retrieve_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: Optional[List[List[float]]] = None,
    top_k: int = 4,
    candidate_pool: int = 12,
):
    """Hybrid (semantic + BM25) retrieval over a wide candidate pool, then a
    feature-based reranking pass (#3) to pick the final top_k. BM25-only
    fallback applies when no embeddings are available."""
    query_embedding = None
    if chunk_embeddings is not None:
        query_vectors = embed_texts([question], task_type="RETRIEVAL_QUERY")
        if query_vectors:
            query_embedding = query_vectors[0]

    candidates = hybrid_rank(
        question,
        chunks,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings if query_embedding is not None else None,
        top_k=max(candidate_pool, top_k),
    )

    return rerank(question, candidates, top_k=top_k)


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    answer_mode: str,
    web_results: Optional[List[Dict[str, str]]] = None,
    history_summary: str = "",
):
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
Document Source {index}
File: {chunk["source"]}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Similarity Score: {chunk["score"]:.3f}
Matched Terms: {", ".join(chunk.get("matched_terms", [])) or "No exact keyword overlap"}
Content:
{chunk["text"]}
"""
        )

    context = "\n".join(context_parts)

    web_context = ""
    if web_results:
        web_context = "\n".join(
            f"""
Web Source {index}
Title: {result.get("title", "Untitled")}
URL: {result.get("url", "")}
Snippet:
{result.get("snippet", "")}
"""
            for index, result in enumerate(web_results, start=1)
        )

    history_block = (
        f"\nConversation summary (for resolving references only):\n{history_summary}\n"
        if history_summary
        else ""
    )

    mode = MODES[resolve_mode(answer_mode)]
    tone_rule = mode["tone"]

    return f"""
You are an AI RAG Research Assistant.

{tone_rule}

Source Rules:
- Treat the injected document context as your primary knowledge base / subject matter.
- The user may ask you to GENERATE or TRANSFORM content based on the documents — e.g.
  "quiz me", "write interview questions and answers", "summarize", "give examples",
  "make flashcards", "compare". For these tasks, USE the document content as the basis
  and produce what was asked. Do NOT refuse just because the exact wording (e.g. a
  literal "interview questions" section) is not present in the documents.
- Only reply that information is missing for a FACTUAL LOOKUP whose specific answer
  genuinely is not in the context (and is not available from the web context either).
- Use the external web context only when it is provided and relevant.
- Use the conversation summary only to understand references, not as a fact source.
- Ground every claim in the documents; cite as [File name, Page X] and web results as
  [Web Source X]. Do not invent facts or sources beyond what the content supports.
- Keep the answer structured and easy to scan.
{history_block}
Injected Document Context:
{context}

External Web Context:
{web_context or "None"}

User Question:
{question}

Answer:
"""


def compute_confidence(
    retrieved_chunks: List[Dict[str, Any]], question: str, semantic_active: bool
) -> Dict[str, Any]:
    """Estimate how well the retrieved context supports an answer, from retrieval
    signals only (no LLM call). Blends semantic similarity with query-term coverage."""
    if not retrieved_chunks:
        return {
            "score": 0.0,
            "label": "low",
            "warning": "No relevant content was retrieved — the answer is unlikely to be grounded in your documents.",
            "semantic_signal": 0.0,
            "lexical_signal": 0.0,
        }

    top = retrieved_chunks[:3]

    sem_vals = [float(c.get("semantic_score") or 0.0) for c in top]
    sem_signal = sum(sem_vals) / len(sem_vals) if sem_vals else 0.0
    # cosine ~0.6 with text-embedding-3-small already indicates a strong match.
    sem_scaled = min(1.0, sem_signal / 0.6)

    query_terms = set(tokenize(question))
    if query_terms:
        covered = set()
        for c in top:
            covered |= set(c.get("matched_terms", []))
        lexical_signal = len(covered & query_terms) / len(query_terms)
    else:
        lexical_signal = 0.0

    if semantic_active:
        score = 0.6 * sem_scaled + 0.4 * lexical_signal
    else:
        score = lexical_signal

    score = round(min(1.0, max(0.0, score)), 3)
    label = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    warning = (
        "Low retrieval confidence — this answer may not be well supported by your documents. "
        "Treat it cautiously, rephrase your question, or upload a more relevant document."
        if label == "low"
        else None
    )

    return {
        "score": score,
        "label": label,
        "warning": warning,
        "semantic_signal": round(sem_scaled, 3),
        "lexical_signal": round(lexical_signal, 3),
    }


def prepare_rag_trace(
    question: str,
    answer_mode: str,
    files_payload: List[Dict[str, Any]],
    use_web: bool = False,
    history_summary: str = "",
    has_history: bool = False,
    source_warnings: Optional[List[Dict[str, str]]] = None,
):
    index = build_document_index(files_payload)
    chunks = index["chunks"]

    route = route_question(question, has_history=has_history, use_web=use_web)
    web_results = web_search(question) if route == "web" else []

    retrieved_chunks = retrieve_chunks(
        question, chunks, chunk_embeddings=index["chunk_embeddings"], top_k=4
    )
    prompt = build_rag_prompt(
        question, retrieved_chunks, answer_mode,
        web_results=web_results, history_summary=history_summary,
    )

    citations = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk.get("section", ""),
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "rerank_score": chunk.get("rerank_score"),
            "rerank_features": chunk.get("rerank_features"),
            "preview": chunk["text"][:700],
            "matched_terms": chunk.get("matched_terms", []),
            "top_terms": chunk.get("top_terms", []),
        }
        for chunk in retrieved_chunks
    ]

    used_semantic = index["semantic"] and any(
        chunk.get("semantic_score") for chunk in retrieved_chunks
    )
    confidence = compute_confidence(retrieved_chunks, question, used_semantic)

    route_labels = {"rag": "Document RAG", "web": "Web Search", "memory": "Conversation Memory"}

    return {
        "question": question,
        "answer_mode": answer_mode,
        "mode": MODES[resolve_mode(answer_mode)]["label"],
        "route": route_labels.get(route, "Document RAG"),
        "web_results": web_results,
        "pages_processed": index["pages"],
        "chunks_processed": len(chunks),
        # Sources skipped before indexing (type/size/URL) + files that failed to extract.
        "warnings": (source_warnings or []) + index.get("warnings", []),
        "retrieved_chunks": citations,
        "confidence": confidence,
        "prompt": prompt,
        "embedding_summary": {
            "method": (
                f"Hybrid retrieval: {EMBEDDING_MODEL} (semantic) + BM25 (lexical), via OpenRouter"
                if used_semantic
                else "BM25 lexical retrieval (semantic embeddings unavailable, fell back)"
            ),
            "model": EMBEDDING_MODEL,
            "query_terms": tokenize(question)[:16],
            "semantic_active": used_semantic,
        },
    }


def generate_answer(prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None):
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens or GENERATION_MAX_TOKENS or NOT_GIVEN,
        extra_body=GENERATION_EXTRA_BODY,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_generation_usage(usage.prompt_tokens, usage.completion_tokens)
    return response.choices[0].message.content or "I could not generate an answer."


def stream_answer(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    usage_sink: Optional[Dict[str, int]] = None,
):
    client = get_openrouter_client()
    stream = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens or GENERATION_MAX_TOKENS or NOT_GIVEN,
        extra_body=GENERATION_EXTRA_BODY,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None and usage_sink is not None:
            usage_sink["prompt_tokens"] = usage.prompt_tokens or 0
            usage_sink["completion_tokens"] = usage.completion_tokens or 0
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def follow_up_questions(
    answer_mode: str = "Research Mode", retrieved_chunks: Optional[List[Dict[str, Any]]] = None
):
    """Budget-safe smart follow-ups: mode-appropriate templates, made document-aware
    by weaving in the top terms from the retrieved chunks. No LLM call."""
    suggestions = list(MODES[resolve_mode(answer_mode)]["followups"])

    topics: List[str] = []
    for chunk in retrieved_chunks or []:
        for term in chunk.get("top_terms", []):
            if term not in topics:
                topics.append(term)

    if topics:
        suggestions.insert(0, f"Tell me more about {topics[0]}.")
        if len(topics) > 1:
            suggestions.append(f"How does {topics[1]} relate to this?")

    return suggestions[:4]


def _record_generation_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]):
    METRICS["generation_prompt_tokens"] += prompt_tokens or 0
    METRICS["generation_completion_tokens"] += completion_tokens or 0


def estimate_cost(embedding_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        embedding_tokens * PRICE_PER_TOKEN["embedding_in"]
        + prompt_tokens * PRICE_PER_TOKEN["generation_in"]
        + completion_tokens * PRICE_PER_TOKEN["generation_out"]
    )


def build_request_metrics(
    latency_ms: float,
    embedding_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    semantic_active: bool,
) -> Dict[str, Any]:
    """Update session counters and return a per-request metrics summary (#10)."""
    cost = estimate_cost(embedding_tokens, prompt_tokens, completion_tokens)

    METRICS["requests"] += 1
    METRICS["semantic_requests" if semantic_active else "fallback_requests"] += 1
    METRICS["total_cost_usd"] += cost

    requests = max(1, int(METRICS["requests"]))
    return {
        "latency_ms": round(latency_ms),
        "embedding_tokens": embedding_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_cost_usd": round(cost, 6),
        "retrieval_mode": "hybrid (semantic + BM25)" if semantic_active else "BM25 fallback",
        "session_requests": requests,
        "session_fallback_rate": round(METRICS["fallback_requests"] / requests, 3),
        "session_total_cost_usd": round(METRICS["total_cost_usd"], 6),
    }


def _persist_metrics(metrics: Dict[str, Any], route: Optional[str], user_id: Optional[str] = None):
    """Best-effort persisted observability (#10). No-op unless DATABASE_URL is set."""
    store.log_metrics(
        {
            "user_id": user_id,
            "latency_ms": metrics.get("latency_ms"),
            "prompt_tokens": metrics.get("prompt_tokens"),
            "completion_tokens": metrics.get("completion_tokens"),
            "embedding_tokens": metrics.get("embedding_tokens"),
            "request_cost_usd": metrics.get("request_cost_usd"),
            "retrieval_mode": metrics.get("retrieval_mode"),
            "route": route,
        }
    )


def build_history_context(history_json: Optional[str]):
    """Parse the client's chat history and return a (context_string, has_history)
    pair for prompt injection. The whole conversation is passed verbatim: no
    window and no summarising of older turns, so follow-ups can reach any detail."""
    if not history_json:
        return "", False
    try:
        turns = json.loads(history_json)
        if not isinstance(turns, list):
            return "", False
    except (json.JSONDecodeError, TypeError):
        return "", False

    history = format_history(turns)
    return ("Conversation so far:\n" + history if history else ""), has_history(turns)


def serialize_event(event: str, data: Dict[str, Any]):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# The single-page UI lives in src/static (index.html, styles.css, app.js); no build step.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _asset_version() -> str:
    """Content hash of the CSS/JS so a deploy never serves stale cached assets."""
    digest = hashlib.sha256()
    for name in ("styles.css", "app.js"):
        digest.update((STATIC_DIR / name).read_bytes())
    return digest.hexdigest()[:10]


INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8").replace(
    "{{ASSET_VERSION}}", _asset_version()
)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_message = request.message.strip()

    if not user_message:
        return ChatResponse(answer="Please enter a question.", follow_up_questions=[])

    prompt = build_rag_prompt(
        question=user_message,
        retrieved_chunks=[],
        answer_mode=request.answer_mode or "Research Mode",
    )

    try:
        answer = generate_answer(prompt)
    except Exception as error:
        answer = f"Model API error: {str(error)}"

    return ChatResponse(
        answer=answer,
        follow_up_questions=follow_up_questions(request.answer_mode or "Research Mode"),
    )


def _as_list(value: Optional[Any]) -> List[str]:
    """Form fields may arrive as one string or a repeated list; normalise both."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


async def gather_sources(
    files: Optional[List[UploadFile]] = None,
    url: Optional[Any] = None,
    youtube_url: Optional[Any] = None,
):
    """Collect knowledge sources from uploaded files (PDF/DOCX/TXT/MD/images),
    web page URLs, and/or YouTube links (#8). Returns (sources, warnings):
    source dicts that build_document_index() understands, plus one warning per
    source that had to be skipped. Only fails if no source is usable at all."""
    sources, warnings = await collect_sources(files, url, youtube_url)

    if not sources:
        raise HTTPException(
            status_code=400,
            detail=(
                _no_readable_sources_message(warnings)
                if warnings
                else "Please upload a document or provide a URL / YouTube link."
            ),
        )

    return sources, warnings


async def collect_sources(
    files: Optional[List[UploadFile]] = None,
    url: Optional[Any] = None,
    youtube_url: Optional[Any] = None,
):
    """Like gather_sources, but never raises: an empty source list is allowed."""
    sources: List[Dict[str, Any]] = []
    warnings: List[Dict[str, str]] = []

    for upload in files or []:
        if not upload.filename:
            continue
        if not upload.filename.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
            warnings.append({
                "source": upload.filename,
                "message": f"unsupported file type. Supported: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}",
            })
            continue
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            warnings.append({
                "source": upload.filename,
                "message": f"larger than the {MAX_UPLOAD_MB:.0f} MB upload limit.",
            })
            continue
        sources.append({"filename": upload.filename, "bytes": data})

    for page_url in _as_list(url):
        try:
            sources.append({"key": page_url, "pages": extract_url(page_url)})
        except Exception as error:
            warnings.append({"source": page_url, "message": f"could not read this web page ({error})."})

    for video_url in _as_list(youtube_url):
        try:
            sources.append({"key": video_url, "pages": extract_youtube(video_url)})
        except Exception as error:
            warnings.append({"source": video_url, "message": f"could not read the transcript ({error})."})

    return sources, warnings


@app.post("/api/research")
async def research(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[List[str]] = Form(None),
    youtube_url: Optional[List[str]] = Form(None),
    use_web: bool = Form(False),
    history: Optional[str] = Form(None),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    started = time.perf_counter()
    embedding_before = METRICS["embedding_tokens"]
    sources, source_warnings = await gather_sources(files, url, youtube_url)
    history_summary, had_history = build_history_context(history)
    trace = prepare_rag_trace(
        clean_question, answer_mode, sources,
        use_web=use_web, history_summary=history_summary, has_history=had_history,
        source_warnings=source_warnings,
    )
    embedding_used = METRICS["embedding_tokens"] - embedding_before

    mode = MODES[resolve_mode(answer_mode)]
    prompt_before = METRICS["generation_prompt_tokens"]
    completion_before = METRICS["generation_completion_tokens"]

    try:
        answer = generate_answer(
            trace["prompt"], temperature=mode["temperature"]
        )
    except Exception as error:
        METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Model API error: {str(error)}") from error

    metrics = build_request_metrics(
        latency_ms=(time.perf_counter() - started) * 1000,
        embedding_tokens=embedding_used,
        prompt_tokens=METRICS["generation_prompt_tokens"] - prompt_before,
        completion_tokens=METRICS["generation_completion_tokens"] - completion_before,
        semantic_active=trace["embedding_summary"]["semantic_active"],
    )
    _persist_metrics(metrics, trace.get("route"))

    return {
        **trace,
        "answer": answer,
        "follow_up_questions": follow_up_questions(answer_mode, trace["retrieved_chunks"]),
        "metrics": metrics,
    }


@app.post("/api/research/stream")
async def research_stream(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[List[str]] = Form(None),
    youtube_url: Optional[List[str]] = Form(None),
    use_web: bool = Form(False),
    history: Optional[str] = Form(None),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    started = time.perf_counter()
    embedding_before = METRICS["embedding_tokens"]
    sources, source_warnings = await gather_sources(files, url, youtube_url)
    history_summary, had_history = build_history_context(history)
    trace = prepare_rag_trace(
        clean_question, answer_mode, sources,
        use_web=use_web, history_summary=history_summary, has_history=had_history,
        source_warnings=source_warnings,
    )
    embedding_used = METRICS["embedding_tokens"] - embedding_before

    mode = MODES[resolve_mode(answer_mode)]
    semantic_active = trace["embedding_summary"]["semantic_active"]

    def event_stream():
        answer_parts = []
        usage_sink: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        yield serialize_event("trace", trace)

        try:
            for text in stream_answer(
                trace["prompt"],
                temperature=mode["temperature"],
                usage_sink=usage_sink,
            ):
                answer_parts.append(text)
                yield serialize_event("chunk", {"text": text})

            _record_generation_usage(usage_sink["prompt_tokens"], usage_sink["completion_tokens"])
            metrics = build_request_metrics(
                latency_ms=(time.perf_counter() - started) * 1000,
                embedding_tokens=embedding_used,
                prompt_tokens=usage_sink["prompt_tokens"],
                completion_tokens=usage_sink["completion_tokens"],
                semantic_active=semantic_active,
            )
            _persist_metrics(metrics, trace.get("route"))
            yield serialize_event(
                "done",
                {
                    "answer": "".join(answer_parts),
                    "follow_up_questions": follow_up_questions(answer_mode, trace["retrieved_chunks"]),
                    "metrics": metrics,
                },
            )
        except Exception as error:
            METRICS["errors"] += 1
            yield serialize_event("error", {"message": f"Model API error: {str(error)}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def document_insights(index: Dict[str, Any], refresh_questions: bool = False) -> Dict[str, Any]:
    """Key topics + suggested questions for an indexed document set, cached on the
    index. One small LLM call (~$0.0003); keyphrase extraction is the free fallback.
    ``refresh_questions`` keeps the topics and asks for five new questions."""
    cached = index.get("insights")
    if cached and not refresh_questions:
        return cached

    texts = [chunk["text"] for chunk in index["chunks"]]
    keyphrases = (cached or {}).get("keyphrases") or extract_keyphrases(texts)
    prompt = build_insights_prompt(
        keyphrases,
        sample_excerpts(texts),
        avoid_questions=(cached or {}).get("questions") if refresh_questions else None,
    )
    try:
        llm_result = parse_insights(
            generate_answer(prompt, temperature=0.8 if refresh_questions else 0.3, max_tokens=600)
        )
    except Exception:
        llm_result = None

    insights = build_insights(keyphrases, llm_result)
    if cached and refresh_questions:
        insights["topics"], insights["source"] = cached["topics"], cached["source"]
    index["insights"] = insights
    return insights


@app.post("/api/sources/analyze")
async def analyze_sources(
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[List[str]] = Form(None),
    youtube_url: Optional[List[str]] = Form(None),
    refresh_questions: bool = Form(False),
):
    """Index the given sources and report on them for the Knowledge panel:
    per-source status (ready/failed) with page and chunk counts, document totals,
    key topics and suggested questions. Never fails because of one bad source;
    an all-failed request still returns 200 with every source marked failed."""
    sources, warnings = await collect_sources(files, url, youtube_url)
    failed = [
        {"input": w["source"], "name": w["source"], "status": "failed", "message": w["message"],
         "pages": 0, "chunks": 0}
        for w in warnings
    ]
    empty = {"pages": 0, "chunks": 0, "topics": [], "questions": [], "insights_source": None,
             "semantic": False}

    if not sources:
        return {"sources": failed, **empty}

    try:
        index = build_document_index(sources)
    except NoReadableSources as error:
        failed += [
            {"input": w["source"], "name": w["source"], "status": "failed", "message": w["message"],
             "pages": 0, "chunks": 0}
            for w in error.warnings
        ]
        return {"sources": failed, **empty}

    extract_failures = {w["source"]: w["message"] for w in index["warnings"]}
    chunk_counts = Counter(chunk["source"] for chunk in index["chunks"])
    report = []
    for source in sources:
        if "bytes" in source:
            given, name = source["filename"], source["filename"]
        else:
            given = source["key"]
            name = source["pages"][0]["source"] if source.get("pages") else source["key"]
        if given in extract_failures:
            report.append({"input": given, "name": name, "status": "failed",
                           "message": extract_failures[given], "pages": 0, "chunks": 0})
        else:
            report.append({"input": given, "name": name, "status": "ready", "message": "",
                           "pages": index.get("source_pages", {}).get(name, 0),
                           "chunks": chunk_counts.get(name, 0)})

    insights = document_insights(index, refresh_questions=refresh_questions)
    return {
        "sources": report + failed,
        "pages": index["pages"],
        "chunks": len(index["chunks"]),
        "topics": insights["topics"],
        "questions": insights["questions"],
        "insights_source": insights["source"],
        "semantic": index["semantic"],
    }


@app.get("/api/metrics")
def metrics_snapshot():
    """Session-level observability counters (#10). Resets on cold start."""
    requests = max(1, int(METRICS["requests"]))
    return {
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in METRICS.items()},
        "fallback_rate": round(METRICS["fallback_requests"] / requests, 3),
        "generation_model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/api/account/status")
def account_status(user=Depends(get_optional_user)):
    """Whether persistent workspaces are configured, and who is signed in (#9)."""
    return {"persistence_enabled": store.enabled, "authenticated": bool(user), "user": user}


@app.get("/api/workspace/documents")
def workspace_documents(user=Depends(get_optional_user)):
    if not store.enabled:
        return {"persistence_enabled": False, "documents": []}
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to view your saved documents.")
    return {"persistence_enabled": True, "documents": store.list_documents(user["id"])}


@app.get("/api/workspace/chats")
def workspace_chats(user=Depends(get_optional_user)):
    if not store.enabled:
        return {"persistence_enabled": False, "chats": []}
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to view your saved chats.")
    return {"persistence_enabled": True, "chats": store.list_chats(user["id"])}


@app.post("/api/workspace/chats")
def save_workspace_chat(payload: ChatSave, user=Depends(get_optional_user)):
    if not store.enabled:
        raise HTTPException(status_code=503, detail="Persistence is not configured.")
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to save chats.")
    saved = store.save_chat(user["id"], payload.title, payload.turns)
    return {"saved": saved}


def _judge(prompt: str) -> str:
    return generate_answer(prompt, temperature=0.0, max_tokens=8)


@app.get("/api/eval")
def run_evaluation(k: int = 4, judge: bool = False):
    """Run the bundled evaluation set (#4). Retrieval metrics (Hit Rate, MRR,
    Recall@K, Precision@K) are free. Pass ?judge=true to also score Faithfulness
    and Answer Relevance with an LLM judge (costs tokens — off by default)."""
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "eval" / "dataset.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation dataset not found.")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    doc_path = root / dataset["document"]
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval document {dataset['document']} not found.")

    index = build_document_index([{"filename": doc_path.name, "bytes": doc_path.read_bytes()}])

    started = time.perf_counter()
    per_query: List[Dict[str, float]] = []
    cases_out: List[Dict[str, Any]] = []
    judged: List[Dict[str, float]] = []

    for case in dataset["cases"]:
        retrieved = retrieve_chunks(
            case["question"], index["chunks"], chunk_embeddings=index["chunk_embeddings"], top_k=k
        )
        texts = [c["text"] for c in retrieved]
        metrics = query_retrieval_metrics(texts, case["expected"], k=k)
        per_query.append(metrics)
        row = {"question": case["question"], "expected": case["expected"], **metrics}

        if judge:
            answer = generate_answer(
                build_rag_prompt(case["question"], retrieved, "Research Mode"),
                temperature=0.2,
                max_tokens=400,
            )
            faith = judge_faithfulness(answer, "\n\n".join(texts), _judge)
            relevance = judge_answer_relevance(case["question"], answer, _judge)
            row["faithfulness"] = faith
            row["answer_relevance"] = relevance
            judged.append({"faithfulness": faith, "answer_relevance": relevance})

        cases_out.append(row)

    result = {
        "document": dataset["document"],
        "k": k,
        "semantic_active": index["semantic"],
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "retrieval_metrics": aggregate_metrics(per_query),
        "judge_used": judge,
        "cases": cases_out,
    }
    if judge and judged:
        result["answer_quality"] = {
            "faithfulness": round(sum(j["faithfulness"] for j in judged) / len(judged), 3),
            "answer_relevance": round(sum(j["answer_relevance"] for j in judged) / len(judged), 3),
        }
    return result
