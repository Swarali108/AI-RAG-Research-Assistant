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
from openai import OpenAI
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
    from src.memory import format_history, has_history, prepare_history
except ImportError:
    from ingestion import (
        SUPPORTED_UPLOAD_EXTENSIONS,
        extract_pages,
        extract_url,
        extract_youtube,
    )
    from router import route_question, web_search
    from memory import format_history, has_history, prepare_history

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
# Generation: cheap + strong for grounded RAG answers.
GENERATION_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
# Embeddings: nearly free, powers hybrid semantic retrieval.
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
# Optional attribution headers shown on OpenRouter dashboards.
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/Swarali108/AI-RAG-Research-Assistant"),
    "X-Title": os.getenv("OPENROUTER_SITE_NAME", "AI RAG Research Assistant"),
}

# --- Cost guardrails (keep a small budget lasting) ---
# Cap answer length so a runaway response can't burn output tokens.
GENERATION_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "800"))
# Skip semantic embeddings for very large documents (fall back to free BM25),
# so a huge PDF can't rack up embedding costs on every question.
MAX_EMBED_CHUNKS = int(os.getenv("MAX_EMBED_CHUNKS", "250"))

# --- Explain-Like modes (#13): pure prompt engineering, no extra LLM cost ---
MODES: Dict[str, Dict[str, Any]] = {
    "research": {
        "label": "Research Mode",
        "temperature": 0.2,
        "max_tokens": GENERATION_MAX_TOKENS,
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
        "max_tokens": GENERATION_MAX_TOKENS,
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
        "max_tokens": GENERATION_MAX_TOKENS,
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
        "max_tokens": 700,
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
        "max_tokens": 900,
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
        "max_tokens": 240,
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
    "generation_out": 0.40 / 1_000_000,
}

# In-process metrics. Resets on cold start (fine for the lite version; Phase 2
# persists these to Supabase). Surfaced via /api/metrics and the RAG Inspector.
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
    """Resolve the Supabase access token (if any) to a user; None when anonymous
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

    try:
        client = get_openrouter_client()
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        usage = getattr(response, "usage", None)
        if usage is not None:
            METRICS["embedding_tokens"] += getattr(usage, "total_tokens", 0) or 0
        return [item.embedding for item in response.data]
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


def build_document_index(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse, chunk, and embed the input sources once, caching by content hash.
    Each source is either an uploaded file ({"filename","bytes"}) or a set of
    pre-extracted pages ({"key","pages"}) from a URL or YouTube transcript."""
    signature = _document_signature(sources)
    cached = _DOCUMENT_CACHE.get(signature)
    if cached is not None:
        return cached

    pages = []
    for source in sources:
        if "bytes" in source:
            pages.extend(extract_pages(source["filename"], source["bytes"]))
        else:
            pages.extend(source.get("pages", []))

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the uploaded PDFs.",
        )

    chunks = chunk_pages(pages)

    # Cost guard: only embed documents up to MAX_EMBED_CHUNKS; larger ones use
    # BM25-only retrieval (free) instead of paying to embed hundreds of chunks.
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
        max_tokens=max_tokens or GENERATION_MAX_TOKENS,
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
        max_tokens=max_tokens or GENERATION_MAX_TOKENS,
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
    """Best-effort persisted observability (#10). No-op unless Supabase is configured."""
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
    """Parse the client's chat history, compress it if long (#5), and return a
    (context_string, has_history) pair for prompt injection. Budget-safe: the
    summarizer only runs for long conversations."""
    if not history_json:
        return "", False
    try:
        turns = json.loads(history_json)
        if not isinstance(turns, list):
            return "", False
    except (json.JSONDecodeError, TypeError):
        return "", False

    summarizer = lambda text: generate_answer(text, temperature=0.1, max_tokens=200)
    prepared = prepare_history(turns, summarizer_fn=summarizer)

    parts = []
    if prepared["summary"]:
        parts.append("Summary of earlier conversation:\n" + prepared["summary"])
    if prepared["recent"]:
        parts.append("Recent turns:\n" + format_history(prepared["recent"]))

    return "\n\n".join(parts), has_history(turns)


def serialize_event(event: str, data: Dict[str, Any]):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI RAG Research Assistant</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #050713;
      --sidebar: #070a18;
      --panel: rgba(13, 18, 36, 0.92);
      --panel-soft: rgba(21, 27, 52, 0.86);
      --line: rgba(181, 198, 255, 0.14);
      --text: #f8f7ff;
      --muted: #aeb5d3;
      --purple: #9b5cff;
      --pink: #f06abf;
      --cyan: #47ead8;
      --blue: #65b8ff;
      --green: #60f3a9;
      --danger: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(145deg, #050713 0%, #071022 46%, #050713 100%);
      color: var(--text);
    }
    button, input, textarea, select { font: inherit; }
    .app-shell {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 360px;
      height: 100vh;
      overflow: hidden;
    }
    .sidebar {
      padding: 22px;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(9, 12, 28, 0.98), rgba(5, 7, 19, 0.98));
      overflow-y: auto;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }
    .spark {
      width: 44px;
      height: 44px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--purple), var(--pink));
      font-size: 1.6rem;
    }
    .brand strong { display: block; font-size: 1.1rem; }
    .brand span, .muted { color: var(--muted); }
    .new-chat, .primary-action {
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 14px 16px;
      color: white;
      background: linear-gradient(135deg, #7438ff, #b936e8);
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 18px 42px rgba(116, 56, 255, 0.25);
    }
    .profile-field {
      margin-bottom: 16px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(13, 18, 36, 0.72);
    }
    .profile-field label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .profile-field input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      color: var(--text);
      background: rgba(5, 7, 19, 0.72);
      outline: none;
    }
    .nav, .library { margin-top: 18px; }
    .nav-item, .library-item, .recent-item {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 11px 12px;
      color: var(--muted);
      background: transparent;
      text-align: left;
    }
    .nav-item.active, .library-item.active, .recent-item.active {
      color: var(--text);
      border-color: rgba(155, 92, 255, 0.28);
      background: linear-gradient(90deg, rgba(155, 92, 255, 0.24), rgba(71, 234, 216, 0.05));
    }
    .section-label {
      margin: 18px 0 10px;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .main {
      padding: 22px 34px 24px;
      min-width: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(1.9rem, 4vw, 2.65rem);
      line-height: 1;
      letter-spacing: 0;
      color: #c896ff;
    }
    .subtitle { margin: 0; font-size: 1rem; color: var(--text); }
    .share {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 13px;
      color: var(--text);
      background: rgba(18, 24, 48, 0.86);
      cursor: pointer;
    }
    .quick-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0;
      flex: 0 0 auto;
    }
    .quick-card, .panel, .inspector-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(14, 19, 40, 0.92), rgba(8, 12, 28, 0.92));
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
    }
    .quick-card { padding: 12px 14px; min-height: 64px; }
    .quick-card strong { display: block; margin-bottom: 5px; font-size: 0.95rem; }
    .quick-card span { color: var(--muted); font-size: 0.84rem; }
    .tab-bar {
      display: inline-grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(9, 13, 29, 0.82);
      margin-bottom: 10px;
      flex: 0 0 auto;
      width: max-content;
    }
    .tab {
      border: 0;
      border-radius: 6px;
      padding: 8px 14px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
    }
    .tab.active {
      color: white;
      background: linear-gradient(135deg, rgba(155, 92, 255, 0.8), rgba(240, 106, 191, 0.66));
    }
    .tab-view { display: none; }
    .tab-view.active {
      min-height: 0;
    }
    #chatTab.tab-view.active {
      display: flex;
      flex: 1 1 auto;
      overflow: hidden;
    }
    #inspectorTab.tab-view.active {
      display: block;
      flex: 1 1 auto;
      overflow-y: auto;
      padding-right: 6px;
    }
    .chat-panel {
      padding: 18px;
      min-height: 0;
      flex: 1 1 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #messages {
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      padding-right: 8px;
      scroll-behavior: smooth;
    }
    .message-row {
      display: flex;
      gap: 14px;
      margin: 20px 0;
    }
    .message-row.user { justify-content: flex-end; }
    .bubble {
      max-width: min(760px, 88%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      line-height: 1.65;
      white-space: pre-wrap;
      background: rgba(13, 18, 38, 0.94);
    }
    .bubble.user {
      background: linear-gradient(135deg, rgba(89, 47, 202, 0.86), rgba(33, 24, 80, 0.9));
    }
    .bot-face {
      width: 48px;
      height: 48px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: linear-gradient(135deg, #3d2ab8, #d64eb2);
      flex: 0 0 auto;
    }
    .mode-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
      border-radius: 999px;
      padding: 7px 10px;
      color: #ffc3ef;
      background: rgba(240, 106, 191, 0.11);
      border: 1px solid rgba(240, 106, 191, 0.22);
      font-weight: 700;
      font-size: 0.9rem;
    }
    .composer {
      margin-top: 20px;
      border: 1px solid rgba(181, 198, 255, 0.2);
      border-radius: 8px;
      padding: 16px;
      background: rgba(8, 12, 28, 0.84);
      flex: 0 0 auto;
    }
    textarea {
      width: 100%;
      min-height: 88px;
      resize: vertical;
      border: 0;
      outline: none;
      color: white;
      background: transparent;
      line-height: 1.5;
    }
    .composer-actions {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
    }
    .mode-toggle {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      width: min(620px, 100%);
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(9, 13, 29, 0.86);
    }
    .mode-button {
      border: 0;
      border-radius: 6px;
      padding: 9px 8px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-size: 0.82rem;
      line-height: 1.2;
    }
    .mode-button small { opacity: 0.75; font-size: 0.72rem; }
    .mode-button.active {
      color: white;
      background: linear-gradient(135deg, rgba(155, 92, 255, 0.86), rgba(240, 106, 191, 0.72));
    }
    .send {
      width: 54px;
      height: 48px;
      border: 0;
      border-radius: 8px;
      color: white;
      background: linear-gradient(135deg, #7c3aed, #b23fe8);
      cursor: pointer;
      font-size: 1.2rem;
    }
    .send:disabled { opacity: 0.6; cursor: wait; }
    .rightbar {
      padding: 22px 22px;
      border-left: 1px solid var(--line);
      background: rgba(5, 7, 19, 0.68);
      overflow-y: auto;
    }
    .doc-card, .side-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(18, 23, 48, 0.94), rgba(8, 12, 28, 0.94));
      margin-bottom: 16px;
    }
    .file-input {
      width: 100%;
      border: 1px dashed rgba(155, 92, 255, 0.42);
      border-radius: 8px;
      padding: 12px;
      color: var(--muted);
      background: rgba(5, 7, 19, 0.52);
    }
    .progress {
      height: 8px;
      border-radius: 999px;
      margin-top: 14px;
      background: linear-gradient(90deg, var(--purple), var(--pink), var(--cyan));
    }
    .insight-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 14px 0;
    }
    .metric {
      padding: 12px;
      border-radius: 8px;
      background: rgba(21, 27, 52, 0.82);
      border: 1px solid var(--line);
    }
    .metric strong { display: block; font-size: 1.4rem; color: #d8c3ff; }
    .metric span { color: var(--muted); font-size: 0.82rem; }
    .topic-list, .question-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .topic {
      border-radius: 999px;
      padding: 8px 11px;
      color: white;
      background: linear-gradient(135deg, rgba(240, 106, 191, 0.66), rgba(71, 234, 216, 0.24));
      font-size: 0.84rem;
    }
    .suggestion {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px;
      color: var(--text);
      background: rgba(21, 27, 52, 0.72);
      text-align: left;
      cursor: pointer;
    }
    .inspector-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    .inspector-panel {
      padding: 18px;
      min-height: 160px;
    }
    .inspector-panel.full { grid-column: 1 / -1; }
    pre {
      overflow: auto;
      max-height: 340px;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(4, 7, 16, 0.92);
      color: #d7e3ff;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    .score-row {
      margin: 12px 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(21, 27, 52, 0.58);
    }
    .score-track {
      height: 7px;
      border-radius: 999px;
      background: rgba(255,255,255,0.09);
      overflow: hidden;
      margin-top: 8px;
    }
    .score-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--purple), var(--cyan));
    }
    .empty-state {
      padding: 34px;
      text-align: center;
      color: var(--muted);
    }
    .sidebar-empty {
      padding: 11px 12px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      font-size: 0.92rem;
      line-height: 1.45;
    }
    .danger { color: var(--danger); }
    .conf-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 2px 0 12px;
      padding: 6px 11px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      border: 1px solid var(--line);
    }
    .conf-high { color: #bdf7d6; background: rgba(96, 243, 169, 0.12); border-color: rgba(96, 243, 169, 0.3); }
    .conf-medium { color: #ffe2b0; background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.32); }
    .conf-low { color: #ffc3cf; background: rgba(251, 113, 133, 0.12); border-color: rgba(251, 113, 133, 0.34); }
    .answer-warning {
      margin: 10px 0 0;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 0.86rem;
      color: #ffc3cf;
      background: rgba(251, 113, 133, 0.1);
      border: 1px solid rgba(251, 113, 133, 0.3);
    }
    .sources { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 10px; }
    .sources > summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--cyan);
      font-size: 0.88rem;
      list-style: none;
    }
    .source-item {
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(8, 12, 28, 0.6);
    }
    .source-item strong { font-size: 0.86rem; }
    .source-item .source-text {
      margin-top: 6px;
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.5;
      max-height: 150px;
      overflow: auto;
      white-space: pre-wrap;
    }
    mark {
      background: rgba(155, 92, 255, 0.42);
      color: #fff;
      border-radius: 3px;
      padding: 0 2px;
    }
    .answer-metrics {
      margin-top: 10px;
      font-size: 0.76rem;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
    }
    .followups-wrap { margin-top: 12px; }
    .followups-wrap .section-label { margin: 0 0 8px; }
    .file-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 10px;
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(8, 12, 28, 0.6);
    }
    .file-name {
      font-size: 0.82rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 200px;
    }
    .file-remove {
      flex: 0 0 auto;
      border: 0;
      background: transparent;
      color: var(--danger);
      cursor: pointer;
      font-size: 0.95rem;
      line-height: 1;
      padding: 2px 6px;
      border-radius: 6px;
    }
    .file-remove:hover { background: rgba(251, 113, 133, 0.16); }
    .file-clear-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 8px;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .file-clear {
      border: 1px solid rgba(251, 113, 133, 0.34);
      background: rgba(251, 113, 133, 0.1);
      color: var(--danger);
      cursor: pointer;
      font-size: 0.76rem;
      padding: 3px 9px;
      border-radius: 999px;
    }
    .file-clear:hover { background: rgba(251, 113, 133, 0.2); }
    @media (max-width: 1180px) {
      .app-shell { grid-template-columns: 280px minmax(0, 1fr); }
      .rightbar { display: none; }
      .quick-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .app-shell { grid-template-columns: 1fr; }
      .sidebar { display: none; }
      .main { padding: 20px; }
      .quick-grid, .inspector-grid { grid-template-columns: 1fr; }
      .composer-actions { flex-direction: column; align-items: stretch; }
      .send { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="spark">✦</div>
        <div>
          <strong>AI Research Assistant</strong>
          <span>Your AI-powered research companion</span>
        </div>
      </div>
      <div class="profile-field">
        <label for="userName">Your name</label>
        <input id="userName" type="text" placeholder="Enter your name" autocomplete="name" />
      </div>
      <button class="new-chat" id="newChat">+ New Chat ✨</button>

      <div class="nav">
        <button class="nav-item active" id="navHome" data-nav="home">⌂ Home</button>
        <button class="nav-item" id="navLibrary" data-nav="library">▣ Library</button>
        <button class="nav-item" id="navRecent" data-nav="recent">◴ Recent Chats</button>
      </div>

      <div class="section-label">Recent Chats</div>
      <div id="recentChats">
        <div class="sidebar-empty">No recent chats yet.</div>
      </div>

      <div class="section-label">Your Library</div>
      <div class="library" id="libraryList">
        <div class="sidebar-empty">Upload PDFs to see them here.</div>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <h1 id="greetingTitle">Hello Researcher!</h1>
          <p class="subtitle">What would you like to research today?</p>
        </div>
        <button class="share" id="shareButton">↗ Share</button>
      </div>

      <div class="quick-grid">
        <div class="quick-card"><strong>✨ Summarize</strong><span>Summarize any document</span></div>
        <div class="quick-card"><strong>🧠 Key takeaways</strong><span>Extract key points</span></div>
        <div class="quick-card"><strong>♙ Compare</strong><span>Compare frameworks</span></div>
        <div class="quick-card"><strong>💡 Suggest questions</strong><span>Get smart suggestions</span></div>
      </div>

      <div class="tab-bar">
        <button class="tab active" data-tab="chat">Chat</button>
        <button class="tab" data-tab="inspector">RAG Inspector</button>
      </div>

      <section id="chatTab" class="tab-view active">
        <div class="panel chat-panel">
          <div id="messages">
            <div class="empty-state">Upload a PDF, ask a question, and I will answer with citations from your document.</div>
          </div>

          <div class="composer">
            <textarea id="question" placeholder="Ask a follow-up question..."></textarea>
            <label style="display:flex;align-items:center;gap:8px;font-size:0.82rem;color:var(--muted);margin:4px 0 10px;cursor:pointer;">
              <input type="checkbox" id="webToggle" /> 🌐 Search the web for current info
            </label>
            <div class="composer-actions">
              <div class="mode-toggle">
                <button class="mode-button active" data-mode="Research Mode">🔬 Research<br><small>Professional</small></button>
                <button class="mode-button" data-mode="Beginner Mode">🌱 Beginner<br><small>Plain & simple</small></button>
                <button class="mode-button" data-mode="Interview Mode">🎯 Interview<br><small>Prep-ready</small></button>
                <button class="mode-button" data-mode="Professor Mode">🎓 Professor<br><small>In-depth</small></button>
                <button class="mode-button" data-mode="30-Second Summary">⚡ 30-Sec<br><small>TL;DR</small></button>
                <button class="mode-button" data-mode="Bestie Mode">✨ Bestie<br><small>Sassy & fun</small></button>
              </div>
              <button class="send" id="askButton">➤</button>
            </div>
          </div>
        </div>
      </section>

      <section id="inspectorTab" class="tab-view">
        <div class="inspector-grid">
          <div class="inspector-panel">
            <h3>Question</h3>
            <pre id="inspectQuestion">Ask a question to populate the inspector.</pre>
          </div>
          <div class="inspector-panel">
            <h3>Embeddings</h3>
            <pre id="inspectEmbeddings">Waiting for query vector terms.</pre>
          </div>
          <div class="inspector-panel full">
            <h3>Retrieved Chunks + Similarity Scores</h3>
            <div id="inspectChunks" class="muted">No retrieved chunks yet.</div>
          </div>
          <div class="inspector-panel full">
            <h3>Prompt Sent To the Model</h3>
            <pre id="inspectPrompt">The full context-injected prompt will appear here.</pre>
          </div>
          <div class="inspector-panel full">
            <h3>Answer</h3>
            <pre id="inspectAnswer">The generated answer will appear here.</pre>
          </div>
        </div>
      </section>
    </main>

    <aside class="rightbar">
      <div class="doc-card">
        <label for="files"><strong id="docTitle">Add knowledge sources</strong></label>
        <input class="file-input" id="files" type="file" accept=".pdf,.docx,.txt,.md,.markdown,.png,.jpg,.jpeg,.webp,.bmp,.tiff" multiple />
        <div class="muted" style="margin-top:8px;font-size:0.78rem;">PDF · DOCX · TXT · Markdown · images (OCR)</div>
        <input id="urlInput" class="profile-field" style="width:100%;margin-top:10px;padding:10px;border-radius:8px;border:1px solid var(--line);background:rgba(5,7,19,0.72);color:var(--text);" type="url" placeholder="…or paste a web page URL" />
        <input id="youtubeInput" class="profile-field" style="width:100%;margin-top:8px;padding:10px;border-radius:8px;border:1px solid var(--line);background:rgba(5,7,19,0.72);color:var(--text);" type="url" placeholder="…or a YouTube link (transcript)" />
        <div class="muted" id="fileStatus" style="margin-top:8px;">No source added yet.</div>
        <div class="progress"></div>
      </div>

      <div class="side-card">
        <h3>Document Insights</h3>
        <div class="insight-grid">
          <div class="metric"><strong id="pageCount">0</strong><span>Pages</span></div>
          <div class="metric"><strong id="chunkCount">0</strong><span>Chunks</span></div>
          <div class="metric"><strong id="topicCount">0</strong><span>Topics</span></div>
        </div>
        <div class="topic-list" id="topics">
          <span class="topic">Governance</span>
          <span class="topic">Ethics</span>
          <span class="topic">Risk</span>
        </div>
      </div>

      <div class="side-card">
        <h3>Suggested Questions</h3>
        <div class="question-list">
          <button class="suggestion">What are the key ideas in this document?</button>
          <button class="suggestion">Summarize this document in simple terms.</button>
          <button class="suggestion">What risks or limitations are mentioned?</button>
          <button class="suggestion">What should I remember for an interview?</button>
        </div>
      </div>

      <div class="side-card">
        <h3>Spill the tea on this doc ☕</h3>
        <p class="muted">Switch to Bestie Mode for TL;DR, vibe check, and source-grounded chaos.</p>
        <button class="primary-action" id="bestieShortcut">Give me the tea! ✨</button>
      </div>
    </aside>
  </div>

  <script>
    const filesInput = document.getElementById("files");
    const fileStatus = document.getElementById("fileStatus");
    const userNameInput = document.getElementById("userName");
    const greetingTitle = document.getElementById("greetingTitle");
    const shareButton = document.getElementById("shareButton");
    const libraryList = document.getElementById("libraryList");
    const recentChats = document.getElementById("recentChats");
    const docTitle = document.getElementById("docTitle");
    const messages = document.getElementById("messages");
    const questionInput = document.getElementById("question");
    const askButton = document.getElementById("askButton");
    const pageCount = document.getElementById("pageCount");
    const chunkCount = document.getElementById("chunkCount");
    const topicCount = document.getElementById("topicCount");
    const topics = document.getElementById("topics");
    const inspectQuestion = document.getElementById("inspectQuestion");
    const inspectEmbeddings = document.getElementById("inspectEmbeddings");
    const inspectChunks = document.getElementById("inspectChunks");
    const inspectPrompt = document.getElementById("inspectPrompt");
    const inspectAnswer = document.getElementById("inspectAnswer");
    let currentMode = "Research Mode";
    let latestAnswer = "";
    let wordQueue = [];
    let streamingTimer = null;
    let activeAnswerNode = null;
    let pendingFinalAnswer = null;
    let chatHistory = [];
    let conversationTurns = [];
    let selectedFiles = [];

    const savedName = localStorage.getItem("ragUserName") || "";
    userNameInput.value = savedName;
    updateGreeting(savedName);

    userNameInput.addEventListener("input", () => {
      const cleanName = userNameInput.value.trim();
      localStorage.setItem("ragUserName", cleanName);
      updateGreeting(cleanName);
    });

    function updateGreeting(name) {
      const displayName = name || "Researcher";
      greetingTitle.textContent = `Hello ${displayName}!`;
    }

    document.querySelectorAll(".tab").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".tab-view").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab + "Tab").classList.add("active");
      });
    });

    document.querySelectorAll(".nav-item").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
        button.classList.add("active");

        if (button.dataset.nav === "home") {
          activateTab("chat");
          questionInput.focus();
        }

        if (button.dataset.nav === "library") {
          filesInput.click();
        }

        if (button.dataset.nav === "recent") {
          recentChats.scrollIntoView({ block: "nearest" });
        }
      });
    });

    shareButton.addEventListener("click", async () => {
      const shareData = {
        title: "AI RAG Research Assistant",
        text: "Try my AI RAG Research Assistant.",
        url: window.location.href
      };

      try {
        if (navigator.share) {
          await navigator.share(shareData);
        } else {
          await navigator.clipboard.writeText(window.location.href);
          flashShareLabel("Copied link");
        }
      } catch (error) {
        if (navigator.clipboard) {
          await navigator.clipboard.writeText(window.location.href);
          flashShareLabel("Copied link");
        }
      }
    });

    function activateTab(tabName) {
      document.querySelectorAll(".tab").forEach(item => {
        item.classList.toggle("active", item.dataset.tab === tabName);
      });
      document.querySelectorAll(".tab-view").forEach(item => item.classList.remove("active"));
      document.getElementById(tabName + "Tab").classList.add("active");
    }

    function flashShareLabel(label) {
      const original = shareButton.textContent;
      shareButton.textContent = label;
      setTimeout(() => {
        shareButton.textContent = original;
      }, 1600);
    }

    document.querySelectorAll(".mode-button").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".mode-button").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        currentMode = button.dataset.mode;
      });
    });

    document.querySelectorAll(".suggestion").forEach(button => {
      button.addEventListener("click", () => {
        questionInput.value = button.textContent.trim();
      });
    });

    document.getElementById("bestieShortcut").addEventListener("click", () => {
      currentMode = "Bestie Mode";
      document.querySelectorAll(".mode-button").forEach(item => {
        item.classList.toggle("active", item.dataset.mode === "Bestie Mode");
      });
      questionInput.value = "Spill the tea on this document. Give me the TL;DR, key points, and why it matters.";
    });

    document.getElementById("newChat").addEventListener("click", () => {
      messages.innerHTML = '<div class="empty-state">Fresh chat ready. Upload a PDF and ask away.</div>';
      conversationTurns = [];
      questionInput.value = "";
      latestAnswer = "";
      updateInspector(null);
    });

    filesInput.addEventListener("change", () => {
      Array.from(filesInput.files || []).forEach(file => {
        const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size);
        if (!isDuplicate) selectedFiles.push(file);
      });
      // Clear so the same file can be re-added later and re-fire change.
      filesInput.value = "";
      renderSelectedFiles();
    });

    function renderSelectedFiles() {
      if (!selectedFiles.length) {
        fileStatus.innerHTML = "No source added yet.";
        docTitle.textContent = "Add knowledge sources";
        libraryList.innerHTML = '<div class="sidebar-empty">Upload files to see them here.</div>';
        return;
      }

      docTitle.textContent = selectedFiles.length === 1 ? selectedFiles[0].name : `${selectedFiles.length} files`;

      const clearAll = selectedFiles.length > 1
        ? '<div class="file-clear-row"><span>' + selectedFiles.length + ' files</span>' +
          '<button type="button" id="clearAllFiles" class="file-clear">✕ Clear all</button></div>'
        : "";

      fileStatus.innerHTML = clearAll + selectedFiles.map((file, index) =>
        '<div class="file-row">' +
        '<span class="file-name" title="' + escapeHtml(file.name) + '">📄 ' + escapeHtml(file.name) + '</span>' +
        '<button type="button" class="file-remove" data-index="' + index + '" title="Remove">✕</button>' +
        '</div>'
      ).join("");

      fileStatus.querySelectorAll(".file-remove").forEach(button => {
        button.addEventListener("click", () => {
          selectedFiles.splice(Number(button.dataset.index), 1);
          renderSelectedFiles();
        });
      });

      const clearAllButton = document.getElementById("clearAllFiles");
      if (clearAllButton) {
        clearAllButton.addEventListener("click", () => {
          selectedFiles = [];
          renderSelectedFiles();
        });
      }

      libraryList.innerHTML = selectedFiles.map(file =>
        '<button class="library-item" type="button">📄 <span>' + escapeHtml(file.name) + '</span></button>'
      ).join("");
    }

    askButton.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        askQuestion();
      }
    });

    async function askQuestion() {
      const files = selectedFiles;
      const urlValue = (document.getElementById("urlInput").value || "").trim();
      const youtubeValue = (document.getElementById("youtubeInput").value || "").trim();
      const question = questionInput.value.trim();

      if (!files.length && !urlValue && !youtubeValue) {
        renderError("Add a source first: upload a file, paste a URL, or add a YouTube link.");
        return;
      }

      if (!question) {
        renderError("Please enter a question.");
        return;
      }

      if (messages.querySelector(".empty-state")) {
        messages.innerHTML = "";
      }

      addMessage("user", question);
      questionInput.value = "";
      updateRecentChats(question);
      const answerNode = addMessage("assistant", "", currentMode);
      latestAnswer = "";
      wordQueue = [];
      pendingFinalAnswer = null;
      activeAnswerNode = answerNode;
      stopWordStreamer();
      askButton.disabled = true;
      inspectAnswer.textContent = "Streaming answer...";

      const formData = new FormData();
      files.forEach(file => formData.append("files", file));
      if (urlValue) formData.append("url", urlValue);
      if (youtubeValue) formData.append("youtube_url", youtubeValue);
      formData.append("question", question);
      formData.append("answer_mode", currentMode);
      formData.append("use_web", document.getElementById("webToggle").checked ? "true" : "false");
      formData.append("history", JSON.stringify(conversationTurns.slice(-10)));
      conversationTurns.push({ role: "user", content: question });

      try {
        const response = await fetch("/api/research/stream", {
          method: "POST",
          body: formData
        });

        if (!response.ok || !response.body) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "Request failed.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();

          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\\n\\n");
          buffer = events.pop() || "";

          for (const eventText of events) {
            handleStreamEvent(eventText, answerNode);
          }
        }

        if (buffer.trim()) {
          handleStreamEvent(buffer, answerNode);
        }
      } catch (error) {
        answerNode.textContent = error.message;
        answerNode.classList.add("danger");
      } finally {
        askButton.disabled = false;
      }
    }

    function handleStreamEvent(eventText, answerNode) {
      const eventLine = eventText.split("\\n").find(line => line.startsWith("event:"));
      const dataLine = eventText.split("\\n").find(line => line.startsWith("data:"));

      if (!eventLine || !dataLine) {
        return;
      }

      const eventName = eventLine.replace("event:", "").trim();
      const payload = JSON.parse(dataLine.replace("data:", "").trim());

      if (eventName === "trace") {
        updateInspector(payload);
        renderAnswerMeta(answerNode, payload);
      }

      if (eventName === "chunk") {
        enqueueWords(payload.text, answerNode);
      }

      if (eventName === "done") {
        pendingFinalAnswer = payload.answer || latestAnswer;
        if (!wordQueue.length) {
          latestAnswer = pendingFinalAnswer;
          answerNode.textContent = latestAnswer;
          inspectAnswer.textContent = latestAnswer;
        }
        renderMetrics(answerNode, payload.metrics);
        renderFollowUps(payload.follow_up_questions);
        showInspectorMetrics(payload.metrics);
        conversationTurns.push({ role: "assistant", content: payload.answer || latestAnswer });
      }

      if (eventName === "error") {
        throw new Error(payload.message || "Streaming failed.");
      }
    }

    function enqueueWords(text, answerNode) {
      const pieces = String(text).match(/\\S+\\s*/g) || [String(text)];
      wordQueue.push(...pieces);
      activeAnswerNode = answerNode;

      if (!streamingTimer) {
        streamingTimer = setInterval(flushNextWord, 26);
      }
    }

    function flushNextWord() {
      if (!wordQueue.length) {
        stopWordStreamer();

        if (pendingFinalAnswer) {
          latestAnswer = pendingFinalAnswer;
          if (activeAnswerNode) {
            activeAnswerNode.textContent = latestAnswer;
          }
          inspectAnswer.textContent = latestAnswer;
        }

        return;
      }

      const nextWord = wordQueue.shift();
      const shouldStickToBottom = isMessagesNearBottom();
      latestAnswer += nextWord;

      if (activeAnswerNode) {
        activeAnswerNode.textContent = latestAnswer;
      }

      inspectAnswer.textContent = latestAnswer;

      if (shouldStickToBottom) {
        scrollMessagesToBottom();
      }
    }

    function stopWordStreamer() {
      if (streamingTimer) {
        clearInterval(streamingTimer);
        streamingTimer = null;
      }
    }

    function addMessage(role, content, mode) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      if (role === "assistant") {
        const face = document.createElement("div");
        face.className = "bot-face";
        face.textContent = "🤖";
        row.appendChild(face);
      }

      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;

      if (role === "assistant") {
        const pill = document.createElement("div");
        pill.className = "mode-pill";
        pill.textContent = modeLabel(mode);
        bubble.appendChild(pill);
        const answer = document.createElement("div");
        answer.textContent = content || "Thinking...";
        bubble.appendChild(answer);
        row.appendChild(bubble);
        messages.appendChild(row);
        scrollMessagesToBottom();
        return answer;
      }

      bubble.textContent = content;
      row.appendChild(bubble);
      messages.appendChild(row);
      scrollMessagesToBottom();
      return bubble;
    }

    function isMessagesNearBottom() {
      return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
    }

    function scrollMessagesToBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function renderError(message) {
      if (messages.querySelector(".empty-state")) {
        messages.innerHTML = "";
      }
      const node = addMessage("assistant", message, currentMode);
      node.classList.add("danger");
    }

    function updateRecentChats(question) {
      chatHistory = [question, ...chatHistory.filter(item => item !== question)].slice(0, 8);
      renderRecentChats();
    }

    function renderRecentChats() {
      if (!chatHistory.length) {
        recentChats.innerHTML = '<div class="sidebar-empty">No recent chats yet.</div>';
        return;
      }

      recentChats.innerHTML = chatHistory.map((item, index) => `
        <button class="recent-item ${index === 0 ? "active" : ""}" type="button" data-question="${escapeHtml(item)}">
          💬 <span>${escapeHtml(shortenText(item, 42))}</span>
        </button>
      `).join("");

      recentChats.querySelectorAll(".recent-item").forEach(button => {
        button.addEventListener("click", () => {
          recentChats.querySelectorAll(".recent-item").forEach(item => item.classList.remove("active"));
          button.classList.add("active");
          questionInput.value = button.dataset.question || "";
          activateTab("chat");
          questionInput.focus();
        });
      });
    }

    function shortenText(text, maxLength) {
      return text.length > maxLength ? text.slice(0, maxLength - 1) + "..." : text;
    }

    function updateInspector(trace) {
      if (!trace) {
        inspectQuestion.textContent = "Ask a question to populate the inspector.";
        inspectEmbeddings.textContent = "Waiting for query vector terms.";
        inspectChunks.textContent = "No retrieved chunks yet.";
        inspectPrompt.textContent = "The full context-injected prompt will appear here.";
        inspectAnswer.textContent = "The generated answer will appear here.";
        pageCount.textContent = "0";
        chunkCount.textContent = "0";
        topicCount.textContent = "0";
        return;
      }

      inspectQuestion.textContent = trace.question;
      inspectEmbeddings.textContent = JSON.stringify(
        { ...trace.embedding_summary, confidence: trace.confidence }, null, 2
      );
      inspectPrompt.textContent = trace.prompt;
      pageCount.textContent = trace.pages_processed;
      chunkCount.textContent = trace.chunks_processed;

      const allTerms = new Set();
      trace.retrieved_chunks.forEach(chunk => {
        (chunk.top_terms || []).slice(0, 3).forEach(term => allTerms.add(term));
      });
      topicCount.textContent = allTerms.size;
      topics.innerHTML = Array.from(allTerms).slice(0, 8).map(term => `<span class="topic">${escapeHtml(term)}</span>`).join("");

      inspectChunks.innerHTML = trace.retrieved_chunks.map((chunk, index) => {
        const percent = Math.max(4, Math.round(Number(chunk.score || 0) * 100));
        return `
          <div class="score-row">
            <strong>Chunk ${index + 1}: ${escapeHtml(chunk.source)} · Page ${chunk.page}</strong>
            <div class="muted">Score ${Number(chunk.score).toFixed(3)} · Matched: ${(chunk.matched_terms || []).map(escapeHtml).join(", ") || "No exact keyword overlap"}</div>
            <div class="score-track"><div class="score-fill" style="width:${percent}%"></div></div>
            <pre>${highlightTerms(chunk.preview, chunk.matched_terms)}</pre>
          </div>
        `;
      }).join("");
    }

    function modeLabel(mode) {
      const map = {
        "Research Mode": "🔬 Research Mode",
        "Beginner Mode": "🌱 Beginner Mode",
        "Interview Mode": "🎯 Interview Mode",
        "Professor Mode": "🎓 Professor Mode",
        "30-Second Summary": "⚡ 30-Second Summary",
        "Bestie Mode": "✨ Bestie Mode"
      };
      return map[mode] || "🔬 Research Mode";
    }

    function highlightTerms(text, terms) {
      let safe = escapeHtml(text);
      const seen = new Set();
      (terms || []).forEach(term => {
        const t = String(term).trim();
        if (t.length < 2) return;
        const key = t.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        // matched_terms/top_terms are alphanumeric tokens, safe to use directly
        safe = safe.replace(new RegExp("(" + t + ")", "gi"), "<mark>$1</mark>");
      });
      return safe;
    }

    function renderAnswerMeta(answerNode, trace) {
      const bubble = answerNode.parentElement;
      if (!bubble || bubble.dataset.metaDone) return;
      bubble.dataset.metaDone = "1";

      const conf = trace.confidence || {};
      const pct = Math.round((Number(conf.score) || 0) * 100);
      const label = conf.label || "low";

      const chip = document.createElement("div");
      chip.className = "conf-chip conf-" + label;
      chip.textContent = "🎯 Confidence: " + pct + "% · " + label;
      if (trace.route) {
        chip.textContent += "  ·  🧭 " + trace.route;
      }
      bubble.insertBefore(chip, answerNode);

      if (conf.warning) {
        const warn = document.createElement("div");
        warn.className = "answer-warning";
        warn.textContent = "⚠️ " + conf.warning;
        bubble.appendChild(warn);
      }

      const chunks = trace.retrieved_chunks || [];
      const web = trace.web_results || [];
      if (chunks.length || web.length) {
        const details = document.createElement("details");
        details.className = "sources";
        const docHtml = chunks.map((c, i) =>
          '<div class="source-item"><strong>[' + (i + 1) + '] ' + escapeHtml(c.source) +
          ' · Page ' + c.page + '</strong> <span class="muted">· score ' +
          Number(c.score).toFixed(3) + (c.section ? ' · ' + escapeHtml(c.section) : '') +
          '</span><div class="source-text">' +
          highlightTerms(c.preview || "", c.matched_terms) + '</div></div>'
        ).join("");
        const webHtml = web.map((w, i) =>
          '<div class="source-item"><strong>🌐 [Web Source ' + (i + 1) + '] ' + escapeHtml(w.title || "") +
          '</strong>' + (w.url ? ' <a href="' + escapeHtml(w.url) + '" target="_blank" rel="noopener" class="muted">link</a>' : '') +
          '<div class="source-text">' + escapeHtml(w.snippet || "") + '</div></div>'
        ).join("");
        details.innerHTML =
          "<summary>📚 Sources (" + (chunks.length + web.length) + ") — exact text used for this answer</summary>" +
          docHtml + webHtml;
        bubble.appendChild(details);
      }
    }

    function renderMetrics(answerNode, metrics) {
      if (!metrics) return;
      const bubble = answerNode.parentElement;
      if (!bubble) return;
      const line = document.createElement("div");
      line.className = "answer-metrics";
      const tokens = (metrics.prompt_tokens || 0) + (metrics.completion_tokens || 0);
      line.innerHTML =
        "<span>⏱ " + metrics.latency_ms + " ms</span>" +
        "<span>🔢 " + tokens + " tokens</span>" +
        "<span>💸 $" + (Number(metrics.request_cost_usd) || 0).toFixed(5) + "</span>" +
        "<span>🔎 " + escapeHtml(metrics.retrieval_mode || "") + "</span>";
      bubble.appendChild(line);
    }

    function renderFollowUps(list) {
      if (!list || !list.length) return;
      const wrap = document.querySelector(".question-list");
      if (!wrap) return;
      wrap.innerHTML = list.map(q => '<button class="suggestion">' + escapeHtml(q) + "</button>").join("");
      wrap.querySelectorAll(".suggestion").forEach(btn => {
        btn.addEventListener("click", () => { questionInput.value = btn.textContent.trim(); });
      });
    }

    function showInspectorMetrics(metrics) {
      if (!metrics || !inspectChunks) return;
      const banner =
        '<div class="score-row"><strong>📊 Request metrics</strong><div class="muted">Latency ' +
        metrics.latency_ms + " ms · Prompt " + metrics.prompt_tokens + " tok · Completion " +
        metrics.completion_tokens + " tok · Embed " + metrics.embedding_tokens + " tok · Cost $" +
        (Number(metrics.request_cost_usd) || 0).toFixed(5) + " · " + escapeHtml(metrics.retrieval_mode || "") +
        " · Session fallback " + Math.round((Number(metrics.session_fallback_rate) || 0) * 100) +
        "%</div></div>";
      inspectChunks.innerHTML = banner + inspectChunks.innerHTML;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
    )


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


async def gather_sources(
    files: Optional[List[UploadFile]] = None,
    url: Optional[str] = None,
    youtube_url: Optional[str] = None,
):
    """Collect knowledge sources from uploaded files (PDF/DOCX/TXT/MD/images),
    a web page URL, and/or a YouTube link (#8). Returns a list of source dicts
    that build_document_index() understands."""
    sources: List[Dict[str, Any]] = []

    for upload in files or []:
        if not upload.filename:
            continue
        if not upload.filename.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename}: unsupported type. Supported: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}",
            )
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} exceeds the {MAX_UPLOAD_MB:.0f} MB upload limit.",
            )
        sources.append({"filename": upload.filename, "bytes": data})

    if url and url.strip():
        try:
            sources.append({"key": url.strip(), "pages": extract_url(url.strip())})
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Could not read URL: {error}") from error

    if youtube_url and youtube_url.strip():
        try:
            sources.append({"key": youtube_url.strip(), "pages": extract_youtube(youtube_url.strip())})
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Could not read YouTube transcript: {error}") from error

    if not sources:
        raise HTTPException(
            status_code=400, detail="Please upload a document or provide a URL / YouTube link."
        )

    return sources


@app.post("/api/research")
async def research(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[str] = Form(None),
    youtube_url: Optional[str] = Form(None),
    use_web: bool = Form(False),
    history: Optional[str] = Form(None),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    started = time.perf_counter()
    embedding_before = METRICS["embedding_tokens"]
    sources = await gather_sources(files, url, youtube_url)
    history_summary, had_history = build_history_context(history)
    trace = prepare_rag_trace(
        clean_question, answer_mode, sources,
        use_web=use_web, history_summary=history_summary, has_history=had_history,
    )
    embedding_used = METRICS["embedding_tokens"] - embedding_before

    mode = MODES[resolve_mode(answer_mode)]
    prompt_before = METRICS["generation_prompt_tokens"]
    completion_before = METRICS["generation_completion_tokens"]

    try:
        answer = generate_answer(
            trace["prompt"], temperature=mode["temperature"], max_tokens=mode["max_tokens"]
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
    url: Optional[str] = Form(None),
    youtube_url: Optional[str] = Form(None),
    use_web: bool = Form(False),
    history: Optional[str] = Form(None),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    started = time.perf_counter()
    embedding_before = METRICS["embedding_tokens"]
    sources = await gather_sources(files, url, youtube_url)
    history_summary, had_history = build_history_context(history)
    trace = prepare_rag_trace(
        clean_question, answer_mode, sources,
        use_web=use_web, history_summary=history_summary, has_history=had_history,
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
                max_tokens=mode["max_tokens"],
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
