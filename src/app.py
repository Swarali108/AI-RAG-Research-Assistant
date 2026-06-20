import hashlib
import json
import os
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

try:
    from src.retrieval import hybrid_rank, tokenize
except ImportError:  # when the app dir itself is on sys.path (some deploy setups)
    from retrieval import hybrid_rank, tokenize


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

EMBEDDING_MODEL = "text-embedding-004"

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


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it in Vercel Environment Variables.")

    return genai.Client(api_key=api_key.strip())


def embed_texts(texts: List[str], task_type: str) -> Optional[List[List[float]]]:
    """Embed texts with Gemini. Returns None on any failure so callers can
    gracefully fall back to lexical-only retrieval."""
    if not texts:
        return []

    try:
        client = get_gemini_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [list(item.values) for item in response.embeddings]
    except Exception:
        return None


def load_pdf_pages(file_bytes: bytes, source_name: str) -> List[Dict[str, Any]]:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "source": source_name,
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def chunk_pages(pages: List[Dict[str, Any]], chunk_words: int = 220, overlap: int = 45):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0
        chunk_index = 1

        while start < len(words):
            end = start + chunk_words
            chunk_text = " ".join(words[start:end])

            if chunk_text:
                chunks.append(
                    {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_id": f"{page['source']}_page_{page['page']}_chunk_{chunk_index}",
                        "text": chunk_text,
                        "top_terms": [term for term, _ in Counter(tokenize(chunk_text)).most_common(8)],
                    }
                )

            if end >= len(words):
                break

            start = max(0, end - overlap)
            chunk_index += 1

    return chunks


def _document_signature(files_payload: List[Dict[str, Any]]) -> str:
    """Stable hash of the uploaded file bytes used as the cache key."""
    hasher = hashlib.sha256()
    for payload in sorted(files_payload, key=lambda item: item["filename"]):
        hasher.update(payload["filename"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(payload["bytes"])
    return hasher.hexdigest()


def build_document_index(files_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse, chunk, and embed the uploaded PDFs once, caching the result by
    content hash. Returns chunks, their embeddings (or None), and page count."""
    signature = _document_signature(files_payload)
    cached = _DOCUMENT_CACHE.get(signature)
    if cached is not None:
        return cached

    pages = []
    for payload in files_payload:
        pages.extend(load_pdf_pages(payload["bytes"], payload["filename"]))

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the uploaded PDFs.",
        )

    chunks = chunk_pages(pages)
    chunk_embeddings = embed_texts(
        [chunk["text"] for chunk in chunks], task_type="RETRIEVAL_DOCUMENT"
    )

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
):
    """Hybrid (semantic + BM25) retrieval, with BM25-only fallback when no
    embeddings are available."""
    query_embedding = None
    if chunk_embeddings is not None:
        query_vectors = embed_texts([question], task_type="RETRIEVAL_QUERY")
        if query_vectors:
            query_embedding = query_vectors[0]

    return hybrid_rank(
        question,
        chunks,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings if query_embedding is not None else None,
        top_k=top_k,
    )


def build_rag_prompt(question: str, retrieved_chunks: List[Dict[str, Any]], answer_mode: str):
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

    if "bestie" in answer_mode.lower() or "friendly" in answer_mode.lower():
        tone_rule = """
Use Bestie Mode.
- Be sassy, sarcastic, fun, and lightly chaotic.
- Use casual internet language naturally: bestie, ngl, lowkey, highkey, fr, respectfully, the tea, main character energy, absolutely cooked.
- Use meme-style framing when it helps: "TL;DR", "the tea", "translation", "tiny reality check".
- Do not overdo slang in every sentence.
- Stay accurate, cite sources, and never invent facts.
- Keep the answer useful first, funny second.
"""
    else:
        tone_rule = """
Use Research Mode.
- Be professional, factual, concise, and recruiter-friendly.
- Use clear structure and citations.
- Avoid slang.
"""

    return f"""
You are an AI RAG Research Assistant.

{tone_rule}

RAG Rules:
- Answer using the injected document context first.
- Include citations when using document information.
- Cite as [File name, Page X].
- Do not invent facts or sources.
- If the answer is not supported by the document context, say that clearly.
- Keep the answer structured and easy to scan.

Injected Document Context:
{context}

User Question:
{question}

Answer:
"""


def prepare_rag_trace(question: str, answer_mode: str, files_payload: List[Dict[str, Any]]):
    index = build_document_index(files_payload)
    chunks = index["chunks"]

    retrieved_chunks = retrieve_chunks(
        question, chunks, chunk_embeddings=index["chunk_embeddings"], top_k=4
    )
    prompt = build_rag_prompt(question, retrieved_chunks, answer_mode)

    citations = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "preview": chunk["text"][:700],
            "matched_terms": chunk.get("matched_terms", []),
            "top_terms": chunk.get("top_terms", []),
        }
        for chunk in retrieved_chunks
    ]

    used_semantic = index["semantic"] and any(
        chunk.get("semantic_score") for chunk in retrieved_chunks
    )

    return {
        "question": question,
        "answer_mode": answer_mode,
        "pages_processed": index["pages"],
        "chunks_processed": len(chunks),
        "retrieved_chunks": citations,
        "prompt": prompt,
        "embedding_summary": {
            "method": (
                "Hybrid retrieval: Gemini text-embedding-004 (semantic) + BM25 (lexical)"
                if used_semantic
                else "BM25 lexical retrieval (semantic embeddings unavailable, fell back)"
            ),
            "model": EMBEDDING_MODEL,
            "query_terms": tokenize(question)[:16],
            "semantic_active": used_semantic,
        },
    }


def generate_answer(prompt: str, temperature: float = 0.2):
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text or "I could not generate an answer."


def stream_answer(prompt: str, temperature: float = 0.2):
    client = get_gemini_client()
    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text


def follow_up_questions(answer_mode: str = "Research Mode"):
    if "bestie" in answer_mode.lower() or "friendly" in answer_mode.lower():
        return [
            "Spill the TL;DR.",
            "Explain this like I am sleep-deprived.",
            "What is the actual tea here?",
        ]

    return [
        "Can you summarize the key points?",
        "What evidence supports this answer?",
        "Can you explain this with an example?",
    ]


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
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
      width: min(520px, 100%);
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(9, 13, 29, 0.86);
    }
    .mode-button {
      border: 0;
      border-radius: 6px;
      padding: 12px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
    }
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
            <div class="composer-actions">
              <div class="mode-toggle">
                <button class="mode-button active" data-mode="Research Mode">🔬 Research Mode<br><small>Professional & factual</small></button>
                <button class="mode-button" data-mode="Bestie Mode">✨ Bestie Mode<br><small>Sassy, fun & chaotic</small></button>
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
            <h3>Prompt Sent To Gemini</h3>
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
        <label for="files"><strong id="docTitle">Upload research PDFs</strong></label>
        <input class="file-input" id="files" type="file" accept="application/pdf" multiple />
        <div class="muted" id="fileStatus">No document uploaded yet.</div>
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
      questionInput.value = "";
      latestAnswer = "";
      updateInspector(null);
    });

    filesInput.addEventListener("change", () => {
      const files = Array.from(filesInput.files || []);
      const names = files.map(file => file.name);
      fileStatus.textContent = files.length ? `${files.length} file(s): ${names.join(", ")}` : "No document uploaded yet.";
      docTitle.textContent = names[0] || "Upload research PDFs";
      renderLibrary(names);
    });

    function renderLibrary(names) {
      if (!names.length) {
        libraryList.innerHTML = '<div class="sidebar-empty">Upload PDFs to see them here.</div>';
        return;
      }

      libraryList.innerHTML = names.map((name, index) => `
        <button class="library-item ${index === 0 ? "active" : ""}" type="button">
          📄 <span>${escapeHtml(name)}</span>
        </button>
      `).join("");
    }

    askButton.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        askQuestion();
      }
    });

    async function askQuestion() {
      const files = Array.from(filesInput.files || []);
      const question = questionInput.value.trim();

      if (!files.length) {
        renderError("Please upload at least one PDF first.");
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
      formData.append("question", question);
      formData.append("answer_mode", currentMode);

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
        pill.textContent = mode === "Bestie Mode" ? "✨ Bestie Mode" : "🔬 Research Mode";
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
      inspectEmbeddings.textContent = JSON.stringify(trace.embedding_summary, null, 2);
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
            <pre>${escapeHtml(chunk.preview)}</pre>
          </div>
        `;
      }).join("");
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
        answer = f"Gemini API error: {str(error)}"

    return ChatResponse(
        answer=answer,
        follow_up_questions=follow_up_questions(request.answer_mode or "Research Mode"),
    )


async def read_uploads(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    payload = []

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF.")

        data = await upload.read()

        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} exceeds the {MAX_UPLOAD_MB:.0f} MB upload limit.",
            )

        payload.append(
            {
                "filename": upload.filename,
                "bytes": data,
            }
        )

    return payload


@app.post("/api/research")
async def research(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: List[UploadFile] = File(...),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    trace = prepare_rag_trace(clean_question, answer_mode, await read_uploads(files))

    try:
        answer = generate_answer(trace["prompt"], temperature=0.55 if "bestie" in answer_mode.lower() else 0.2)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(error)}") from error

    return {
        **trace,
        "answer": answer,
        "follow_up_questions": follow_up_questions(answer_mode),
    }


@app.post("/api/research/stream")
async def research_stream(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: List[UploadFile] = File(...),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    trace = prepare_rag_trace(clean_question, answer_mode, await read_uploads(files))
    temperature = 0.65 if "bestie" in answer_mode.lower() else 0.2

    def event_stream():
        answer_parts = []
        safe_trace = {key: value for key, value in trace.items() if key != "prompt"}
        safe_trace["prompt"] = trace["prompt"]

        yield serialize_event("trace", safe_trace)

        try:
            for text in stream_answer(trace["prompt"], temperature=temperature):
                answer_parts.append(text)
                yield serialize_event("chunk", {"text": text})

            yield serialize_event("done", {"answer": "".join(answer_parts)})
        except Exception as error:
            yield serialize_event("error", {"message": f"Gemini API error: {str(error)}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
