import html
import math
import os
import re
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader


load_dotenv()

app = FastAPI(
    title="AI RAG Research Assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOP_WORDS]


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
                    }
                )

            if end >= len(words):
                break

            start = max(0, end - overlap)
            chunk_index += 1

    return chunks


def lexical_score(question: str, chunk_text: str) -> float:
    query_terms = Counter(tokenize(question))
    chunk_terms = Counter(tokenize(chunk_text))

    if not query_terms or not chunk_terms:
        return 0.0

    overlap = set(query_terms) & set(chunk_terms)
    dot = sum(query_terms[word] * chunk_terms[word] for word in overlap)
    query_norm = math.sqrt(sum(value * value for value in query_terms.values()))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_terms.values()))

    if not query_norm or not chunk_norm:
        return 0.0

    return dot / (query_norm * chunk_norm)


def retrieve_chunks(question: str, chunks: List[Dict[str, Any]], top_k: int = 4):
    scored_chunks = []

    for chunk in chunks:
        score = lexical_score(question, chunk["text"])
        scored_chunks.append({**chunk, "score": score})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    selected = [chunk for chunk in scored_chunks[:top_k] if chunk["score"] > 0]

    if not selected:
        selected = scored_chunks[: min(top_k, len(scored_chunks))]

    return selected


def build_rag_prompt(question: str, retrieved_chunks: List[Dict[str, Any]], answer_mode: str):
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
Document Source {index}
File: {chunk["source"]}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Retrieval Score: {chunk["score"]:.3f}
Content:
{chunk["text"]}
"""
        )

    context = "\n".join(context_parts)

    tone_rule = (
        "Use a friendly, simple explanation style, but keep facts accurate."
        if "gen" in answer_mode.lower()
        else "Use a professional research-assistant tone."
    )

    return f"""
You are an AI RAG Research Assistant.

{tone_rule}

Rules:
- Answer using the provided document context first.
- Include citations when using document information.
- Cite as [File name, Page X].
- Do not invent facts or sources.
- If the answer is not supported by the document context, say that clearly.
- Keep the answer structured and useful.

Document Context:
{context}

User Question:
{question}

Answer:
"""


def generate_answer(prompt: str, temperature: float = 0.2):
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text or "I could not generate an answer."


def follow_up_questions():
    return [
        "Can you summarize the key points?",
        "What evidence supports this answer?",
        "Can you explain this with an example?",
    ]


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
      --bg: #09090b;
      --panel: #111114;
      --panel-2: #18181b;
      --text: #fafafa;
      --muted: #b8b8c2;
      --line: rgba(255,255,255,0.12);
      --accent: #00d6c9;
      --accent-2: #8b5cf6;
      --danger: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #070708;
      padding: 24px;
    }
    main {
      padding: 28px;
      max-width: 1180px;
      width: 100%;
      margin: 0 auto;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.7rem);
      line-height: 1;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 12px 0 24px;
      color: var(--muted);
      font-size: 1.05rem;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.92rem;
      font-weight: 650;
    }
    input, textarea, select, button {
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
    }
    input, textarea, select { padding: 12px; }
    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.5;
    }
    button {
      cursor: pointer;
      margin-top: 12px;
      padding: 12px 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      border: 0;
      color: #050507;
      font-weight: 800;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
    }
    .stat span {
      display: block;
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 6px;
    }
    .stat strong {
      font-size: 1.45rem;
    }
    .answer {
      min-height: 260px;
      white-space: pre-wrap;
      line-height: 1.65;
    }
    .muted { color: var(--muted); }
    .error { color: var(--danger); }
    .citation {
      border-top: 1px solid var(--line);
      padding: 12px 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .citation:first-child { border-top: 0; }
    @media (max-width: 840px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { padding: 20px; }
      .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h2>Document Setup</h2>
      <p class="muted">Upload one or more PDFs, ask a question, and get a source-grounded answer with citations.</p>
      <div class="panel">
        <label for="files">PDF files</label>
        <input id="files" type="file" accept="application/pdf" multiple />
      </div>
      <div class="panel">
        <label for="mode">Answer mode</label>
        <select id="mode">
          <option>Research Mode</option>
          <option>Friendly Mode</option>
        </select>
      </div>
      <div class="panel">
        <div class="muted" id="fileStatus">No PDFs selected yet.</div>
      </div>
    </aside>
    <main>
      <h1>AI RAG Research Assistant</h1>
      <p class="subtitle">A lightweight Vercel deployment that reads PDFs, retrieves relevant context, and answers with Gemini.</p>

      <div class="stats">
        <div class="stat"><span>Documents</span><strong id="docCount">0</strong></div>
        <div class="stat"><span>Chunks Used</span><strong id="chunkCount">0</strong></div>
        <div class="stat"><span>Status</span><strong id="status">Ready</strong></div>
      </div>

      <div class="panel">
        <label for="question">Question</label>
        <textarea id="question" placeholder="Ask something about your uploaded PDF..."></textarea>
        <button id="askButton">Ask Document</button>
      </div>

      <div class="panel answer" id="answer">
        Upload a PDF and ask a question to start.
      </div>

      <div class="panel">
        <h3>Citations</h3>
        <div id="citations" class="muted">Citations will appear here after an answer is generated.</div>
      </div>
    </main>
  </div>

  <script>
    const fileInput = document.getElementById("files");
    const fileStatus = document.getElementById("fileStatus");
    const docCount = document.getElementById("docCount");
    const chunkCount = document.getElementById("chunkCount");
    const statusText = document.getElementById("status");
    const askButton = document.getElementById("askButton");
    const answerBox = document.getElementById("answer");
    const citationsBox = document.getElementById("citations");

    fileInput.addEventListener("change", () => {
      const files = Array.from(fileInput.files || []);
      docCount.textContent = files.length;
      fileStatus.textContent = files.length
        ? files.map(file => file.name).join(", ")
        : "No PDFs selected yet.";
    });

    askButton.addEventListener("click", async () => {
      const files = Array.from(fileInput.files || []);
      const question = document.getElementById("question").value.trim();
      const mode = document.getElementById("mode").value;

      if (!files.length) {
        answerBox.innerHTML = "<span class='error'>Please upload at least one PDF first.</span>";
        return;
      }

      if (!question) {
        answerBox.innerHTML = "<span class='error'>Please enter a question.</span>";
        return;
      }

      const formData = new FormData();
      files.forEach(file => formData.append("files", file));
      formData.append("question", question);
      formData.append("answer_mode", mode);

      askButton.disabled = true;
      statusText.textContent = "Thinking";
      answerBox.textContent = "Reading PDF, retrieving context, and generating answer...";
      citationsBox.textContent = "Waiting for citations...";

      try {
        const response = await fetch("/api/research", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Request failed.");
        }

        answerBox.textContent = data.answer;
        chunkCount.textContent = data.citations.length;

        citationsBox.innerHTML = data.citations.map(item => `
          <div class="citation">
            <strong>${escapeHtml(item.source)}</strong><br />
            Page ${item.page} | Score ${Number(item.score).toFixed(3)}
          </div>
        `).join("") || "No citations returned.";

        statusText.textContent = "Ready";
      } catch (error) {
        answerBox.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
        citationsBox.textContent = "No citations returned.";
        statusText.textContent = "Error";
      } finally {
        askButton.disabled = false;
      }
    });

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

    history_text = ""

    for message in request.history[-6:]:
        role = message.get("role", "user")
        content = message.get("content", "")
        history_text += f"{role.upper()}: {content}\n"

    prompt = f"""
You are an AI Research Assistant.

Answer clearly, accurately, and professionally.
If you do not know something, say so instead of inventing facts.

Conversation history:
{history_text}

User question:
{user_message}

Answer:
"""

    try:
        answer = generate_answer(prompt)
    except Exception as error:
        answer = f"Gemini API error: {str(error)}"

    return ChatResponse(answer=answer, follow_up_questions=follow_up_questions())


@app.post("/api/research")
async def research(
    question: str = Form(...),
    answer_mode: str = Form("Research Mode"),
    files: List[UploadFile] = File(...),
):
    clean_question = question.strip()

    if not clean_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    pages = []

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF.")

        file_bytes = await upload.read()

        try:
            pages.extend(load_pdf_pages(file_bytes, upload.filename))
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read {upload.filename}: {str(error)}",
            ) from error

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the uploaded PDFs.",
        )

    chunks = chunk_pages(pages)
    retrieved_chunks = retrieve_chunks(clean_question, chunks, top_k=4)
    prompt = build_rag_prompt(clean_question, retrieved_chunks, answer_mode)

    try:
        answer = generate_answer(prompt)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(error)}") from error

    citations = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "preview": html.escape(chunk["text"][:240]),
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "citations": citations,
        "chunks_processed": len(chunks),
        "pages_processed": len(pages),
        "follow_up_questions": follow_up_questions(),
    }
