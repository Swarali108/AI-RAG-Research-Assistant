# Legacy prototype (not used by the app)

This folder holds an earlier version of the pipeline, kept for reference only.

- `rag.py` — answer generation with Google Gemini (`google-genai`) and a Streamlit UI.
- `embedding.py` — local embeddings with `sentence-transformers` (MiniLM).
- `manual_*_check.py` — hand-run scripts that exercised that version. They are
  not part of the pytest suite and CI does not run them.

The live app is `src/app.py`: generation via OpenRouter, embeddings via
`openai/text-embedding-3-small`, hybrid retrieval in `src/retrieval.py`.
Nothing in `src/` or `tests/` imports this folder.

The scripts still import `src/pdf_loader.py`, `src/chunking.py` and
`src/vector_store.py`, which also belong to this old version. Running them
needs packages that `requirements.txt` no longer installs, such as
`google-genai`, `sentence-transformers` and `faiss-cpu`.
