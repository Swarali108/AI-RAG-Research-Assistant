from pathlib import Path

files = {
    'PRD.md': """# Product Requirements Document

## Product

AI Research Assistant for document-based question answering.

## Problem

Large PDF documents are hard to search using keywords alone. Users need a way to ask questions and receive grounded answers with citations.

## Goal

Build a PDF QA system that:

- Accepts PDF uploads
- Extracts text reliably
- Chunks content for retrieval
- Generates embeddings for semantic search
- Stores vectors locally in FAISS
- Retrieves relevant context for queries
- Sends context to Gemini
- Returns answers with source citations

## Target Users

- Students
- Researchers
- Engineers
- Interview prep candidates

## MVP Features

- PDF upload
- Text extraction
- Fixed and recursive chunking
- Embedding generation
- FAISS vector store
- Similarity search
- Gemini answer generation
- Citation metadata

## Future Enhancements

- Agentic RAG decision path
- Governance warnings
- Audit logging
- AWS deployment
- Semantic chunking

## Success Metrics

- Accurate retrieval
- Source-grounded responses
- Response time under 5 seconds
- Clean documentation and repo structure
""",
    'LLD.md': """# Low-Level Design

## Architecture

- PDF upload UI → PDF loader → chunking → embedding service → FAISS vector store → retriever → Gemini LLM → answer display

## Components

### src/pdf_loader.py

- `extract_text_from_pdf(pdf_path)`
- Returns raw text from PDF pages

### src/chunking.py

- `fixed_size_chunks(text, chunk_size, overlap)`
- `recursive_chunk_text(text, chunk_size, overlap)`
- Keep chunks under model context limits while preserving meaning

### src/embedding.py

- `get_embedding_model()`
- `embed_texts(texts)`
- Uses `all-MiniLM-L6-v2`

### src/vector_store.py

- `FaissVectorStore`
- Stores embeddings, text, and metadata
- Supports `add` and `search`

### src/prompts.py

- Prompt template for grounded generation
- Includes context and question

### src/rag.py

- `search_documents(query, vector_store, top_k)`
- `generate_answer(question, chunks)`
- Uses Gemini via `google.generativeai`

### src/app.py

- Streamlit UI for upload, indexing, and question answering
- Upload PDF, build index, ask question, show answer + sources

## Data Flow

1. User uploads PDF
2. Extract text
3. Split into chunks
4. Create embeddings
5. Build FAISS index
6. Query index for top-k chunks
7. Pass chunks + question to Gemini
8. Render answer and citations

## Notes

- Use environment variable `GENAI_API_KEY`
- Keep the model simple initially
- Add audit logs and warnings later
""",
    'README.md': """# AI RAG Research Assistant

An AI Research Assistant that turns PDF documents into a retrieval-augmented QA system.

## What it does

- Upload PDFs
- Extract document text
- Split text into searchable chunks
- Generate semantic embeddings
- Store vectors in FAISS
- Retrieve relevant passages
- Send context to Gemini for grounded answers
- Display citations and source references

## Tech stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini (via `google-generativeai`)
- PDF extraction with `pypdf`

## Setup

1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Set `GENAI_API_KEY` in your environment
4. Run: `streamlit run src/app.py`

## Status

- Basic repo scaffold created
- Core module stubs implemented
- Next step: add document upload, chunking, embedding, vector search, and Gemini integration
""",
    'src/pdf_loader.py': """import os
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages).strip()
""",
    'src/chunking.py': """import re
from typing import List


def fixed_size_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    clean = text.strip().replace("\n", " ")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def recursive_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\\s+", paragraph)
        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) + 1 <= chunk_size:
                buffer = f"{buffer} {sentence}".strip()
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sentence

        if buffer:
            chunks.append(buffer)

    if overlap > 0 and len(chunks) > 1:
        merged = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged.append(chunk)
                continue
            prev = merged[-1]
            overlap_text = prev[-overlap:].strip()
            merged.append(f"{overlap_text} {chunk}".strip())
        return merged

    return chunks
""",
    'src/embedding.py': """import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

_model = None


def get_embedding_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms
""",
    'src/vector_store.py': """import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embeddings must be a 2D array with the correct dimension.")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, ids = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []

        for score_row, id_row in zip(scores, ids):
            for score, idx in zip(score_row, id_row):
                if idx == -1:
                    continue
                results.append((self.texts[idx], self.metadatas[idx], float(score)))

        return results


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
""",
    'src/prompts.py': """PROMPT_TEMPLATE = """You are an expert research assistant.

Use the extracted context to answer the question as accurately as possible.
Cite the source page or document when available.

Context:
{context}

Question: {question}

Answer:
"""
""",
    'src/rag.py': """import os
from typing import List
import google.generativeai as genai

from .prompts import PROMPT_TEMPLATE


genai.configure(api_key=os.getenv("GENAI_API_KEY", ""))


def build_context(chunks: List[dict], top_k: int = 5) -> str:
    selected = chunks[:top_k]
    context_parts = []
    for item in selected:
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown source")
        context_parts.append(f"[{source}] {item['text']}")
    return "\n\n".join(context_parts)


def generate_answer(question: str, retrieved: List[dict], model: str = "gemini-2.5-flash", temperature: float = 0.2) -> str:
    if not question.strip():
        return "Please provide a question."

    context = build_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question.strip())

    if not os.getenv("GENAI_API_KEY"):
        raise EnvironmentError("GENAI_API_KEY is not set. Configure Google Gemini credentials before running.")

    response = genai.generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=512,
    )

    return response.text.strip()


def search_and_answer(question: str, store, query_embedding, top_k: int = 5) -> str:
    results = store.search(query_embedding, top_k=top_k)
    retrieved = [
        {"text": text, "metadata": metadata, "score": score}
        for text, metadata, score in results
    ]
    return generate_answer(question, retrieved)
""",
    'src/app.py': """import os
import tempfile
from typing import List

import streamlit as st

from pdf_loader import extract_text_from_pdf
from chunking import fixed_size_chunks, recursive_chunk_text
from embedding import embed_texts, normalize_embeddings
from vector_store import FaissVectorStore
from rag import search_and_answer


st.set_page_config(page_title=\"AI RAG Research Assistant\", layout=\"wide\")


@st.cache_data(show_spinner=False)
def build_chunks(text: str, method: str = \"fixed\") -> List[str]:
    if method == \"recursive\":
        return recursive_chunk_text(text, chunk_size=900, overlap=150)
    return fixed_size_chunks(text, chunk_size=900, overlap=150)


@st.cache_data(show_spinner=False)
def build_vector_store(chunks: List[str]):
    embeddings = embed_texts(chunks)
    embeddings = normalize_embeddings(embeddings)
    store = FaissVectorStore(dimension=embeddings.shape[1])
    metadata = [{\"source\": f\"chunk_{i + 1}\"} for i in range(len(chunks))]
    store.add(embeddings, chunks, metadata)
    return store


def main():
    st.title(\"AI RAG Research Assistant\")
    st.write(\"Upload a PDF, build a vector index, and ask questions with citations.\")

    uploaded_file = st.file_uploader(\"Upload a PDF document\", type=[\"pdf\"])
    chunk_method = st.radio(\"Chunking strategy\", [\"fixed\", \"recursive\"], index=0)

    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=\".pdf\") as tmp_file:
            tmp_file.write(bytes_data)
            tmp_path = tmp_file.name

        text = extract_text_from_pdf(tmp_path)

        if not text:
            st.error(\"Unable to extract text from PDF. Try a different file.\")
            return

        if st.button(\"Build index\"):
            with st.spinner(\"Creating chunks and embeddings...\"):
                chunks = build_chunks(text, method=chunk_method)
                store = build_vector_store(chunks)
                st.session_state[\"rag_store\"] = store
                st.session_state[\"chunk_method\"] = chunk_method
                st.success(f\"Index built with {len(chunks)} chunks.\")

    if \"rag_store\" in st.session_state:
        query = st.text_input(\"Ask a question about the uploaded PDF\")
        if query and st.button(\"Get answer\"):
            with st.spinner(\"Retrieving context and generating answer...\"):
                query_embedding = normalize_embeddings(embed_texts([query]))
                answer = search_and_answer(query, st.session_state[\"rag_store\"], query_embedding, top_k=5)
                st.markdown(\"### Answer\")
                st.write(answer)
                st.success(\"Generated with source-grounded context.\")

    st.sidebar.header(\"Setup\")
    st.sidebar.write(\"Set `GENAI_API_KEY` before running the app.\")


if __name__ == \"__main__\":
    main()
""",
    'tests/test_chunking.py': """from src.chunking import fixed_size_chunks, recursive_chunk_text


def test_fixed_size_chunks():
    text = \"This is a test sentence. \" * 50
    chunks = fixed_size_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_recursive_chunk_text():
    text = \"\n\n\".join([\"This is sentence one. This is sentence two.\"] * 5)
    chunks = recursive_chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
""",
}

for relative_path, content in files.items():
    destination = Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
