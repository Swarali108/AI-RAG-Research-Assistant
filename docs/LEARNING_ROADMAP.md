# 📚 Learning Roadmap — AI RAG Research Assistant (zero → expert)

A complete map of everything in and around this project. Work top-to-bottom: each
part lists **topics → sub-topics → questions you should be able to answer**, with
pointers to the exact file in your repo. Treat every ❓ as a flashcard.

> **How to use this:** pick a part, read around the questions (Google / the linked file),
> then explain the answer out loud as if teaching a friend. If you can't, you don't know it yet.
> ✅ = "I can explain this without notes."

**Suggested order:** Part 1 → 2 → 3 → 4 (the big ideas) → 6 (Python) → 5 (your architecture)
→ 7–10 (backend/frontend/math/data) → 11 (DSA) → 12–16 (production & theory).

---

## Part 1 — Absolute foundations (what even is this?)

**1.1 Software & the web**
- ❓ What is a program? What is source code vs. a running app?
- ❓ What is a "web app"? How is it different from a desktop app?
- ❓ What are **client** and **server**? Which part is the browser, which is your `app.py`?
- ❓ What is **frontend** vs **backend**? In this repo, both live in [`src/app.py`](../src/app.py) — how?
- ❓ What is HTTP? What are requests and responses? What is a **status code** (200, 400, 404, 500)?
- ❓ What is a URL? What is a domain (your `…vercel.app`)?

**1.2 How your app is shaped**
- ❓ What is a "single-page app served by the backend"? Why does your FastAPI return a big HTML string?
- ❓ What is a "RAG pipeline" in one sentence?
- ❓ Draw the flow: upload → extract → chunk → embed → retrieve → rerank → prompt → LLM → answer. (See your README diagram.)

---

## Part 2 — APIs, keys, secrets, environments

**2.1 APIs**
- ❓ What is an **API** (Application Programming Interface)?
- ❓ What is an API **endpoint**? (e.g. your `/api/research`, `/api/metrics`.)
- ❓ What are the HTTP **methods** GET vs POST? Which do your endpoints use and why?
- ❓ What is JSON? Why is it the format for API data?
- ❓ What does "OpenAI-compatible API" mean, and why does it let you point the `openai` library at OpenRouter?

**2.2 API keys & auth**
- ❓ What is an **API key**? (Your example.) Why does a service need one?
- ❓ What is the difference between **authentication** (who you are) and **authorization** (what you can do)?
- ❓ What is a **Bearer token**? (Your `Authorization: Bearer <token>` for Supabase.)
- ❓ Why is `OPENROUTER_API_KEY` secret, but the publishable Supabase key is "less" secret?
- ❓ Service-role key vs anon/publishable key — which goes on the backend and why? (You hit this bug!)

**2.3 Environments & config**
- ❓ What is an **environment variable**? Why not hard-code keys in the code?
- ❓ What is the `.env` file? Why is it in `.gitignore`? What does `python-dotenv` / `load_dotenv()` do?
- ❓ Why do you set the same keys in Vercel's dashboard separately?
- ❓ In code: `os.getenv("OPENROUTER_API_KEY")` — what does this do? (See [`src/app.py`](../src/app.py).)

---

## Part 3 — LLMs, tokens, embeddings, models

**3.1 What is an LLM**
- ❓ What is an **LLM** (Large Language Model)? What does "next-token prediction" mean?
- ❓ What is a **token**? Why is text measured in tokens, not words/characters?
- ❓ What is a **context window**? Why does it limit how much you can stuff into a prompt?
- ❓ What is **temperature**? Why do you set `0.2` for Research mode but `0.65` for Bestie? (See `MODES`.)
- ❓ What is `max_tokens`, and how does it protect your $1 budget? (See `GENERATION_MAX_TOKENS`.)
- ❓ What is a **system prompt** vs a **user message**? (Your `messages=[{"role":"user",...}]`.)
- ❓ What is **streaming** and why does it feel faster? (Your `stream=True` + SSE.)
- ❓ What is a **hallucination**? How does RAG + citations reduce it?

**3.2 Providers & models (why more than one)**
- ❓ Who are the model **providers** (OpenAI, Google/Gemini, Anthropic, Meta)?
- ❓ What is **OpenRouter** and why did you route through it instead of calling Gemini directly?
- ❓ Why use **different models for different jobs**? You use:
  - `google/gemini-2.5-flash-lite` for **generation** (cheap, good at writing answers).
  - `openai/text-embedding-3-small` for **embeddings** (turns text into vectors, near-free).
- ❓ What's the trade-off between a "flash/mini/lite" model and a big "pro" model (cost vs quality vs speed)?
- ❓ How is **pricing per million tokens** calculated? (See `PRICE_PER_TOKEN` in your code.)

**3.3 Embeddings (the heart of semantic search)**
- ❓ What is an **embedding**? Why is "a list of 1536 numbers" a *meaning* of the text?
- ❓ Why do similar meanings have vectors that are "close"? What is vector space?
- ❓ What is the difference between embedding a **document** vs a **query** (task types)?
- ❓ Why 1536 dimensions? (It's `text-embedding-3-small`'s output size — matches your pgvector column.)

---

## Part 4 — RAG (Retrieval-Augmented Generation) in depth

**4.1 The concept**
- ❓ What problem does RAG solve that a plain chatbot can't? (Grounding in *your* docs, fresh data, citations.)
- ❓ What are the stages of RAG? (Ingest → chunk → embed → store → retrieve → [rerank] → generate.)
- ❓ Why "retrieve then generate" instead of "fine-tune the model on my docs"?

**4.2 Types of retrieval (and what you chose)**
- ❓ **Lexical / keyword** search (BM25, TF-IDF) — matches exact words. Strength? Weakness?
- ❓ **Semantic / vector** search (embeddings + cosine) — matches meaning. Strength? Weakness?
- ❓ **Hybrid** search = lexical + semantic combined. **This is what you use.** Why is it better than either alone? (Example: "downsides" vs "limitations".)
- ❓ What is **reranking** and why add it after retrieval? (See [`src/reranker.py`](../src/reranker.py).)
- ❓ What is **graph RAG** / **agentic RAG** (you added an agentic router — Part 5.6)?
- ❓ What is **top-k**? Why retrieve a wide pool then rerank to a small k?

**4.3 Chunking**
- ❓ Why split documents into **chunks** instead of embedding the whole file?
- ❓ What is **chunk overlap** and why does it prevent losing context at boundaries?
- ❓ What is **fixed-size** vs **heading-aware** chunking? Which did you implement and why? (`chunk_pages`, `_split_sections`.)

**4.4 Generation & grounding**
- ❓ How do retrieved chunks get **injected into the prompt**? (See `build_rag_prompt`.)
- ❓ What is a **citation** and how do you make the model produce `[File, Page X]`?
- ❓ What is a **confidence score** here, and why is it computed from retrieval signals, not the LLM? (`compute_confidence`.)
- ❓ Why did your app over-refuse generative tasks, and how did the prompt fix it? (Factual lookup vs generate-from-docs.)

---

## Part 5 — YOUR architecture, module by module

For each file: ❓ What is its single responsibility? ❓ What does it import? ❓ What would break if you deleted it?

- **5.1** [`src/app.py`](../src/app.py) — FastAPI app, routes, the HTML/JS UI, orchestration, metrics.
- **5.2** [`src/retrieval.py`](../src/retrieval.py) — `tokenize`, `bm25_scores`, `cosine_similarity`, `hybrid_rank`. ❓ Why is this "pure" (no API calls) and why does that make it testable?
- **5.3** [`src/reranker.py`](../src/reranker.py) — feature-based reranking (phrase/coverage/heading/proximity).
- **5.4** [`src/ingestion.py`](../src/ingestion.py) — multi-format loaders (PDF/DOCX/TXT/MD/URL/YouTube/OCR). ❓ Why lazy imports + graceful errors?
- **5.5** [`src/evaluation.py`](../src/evaluation.py) — retrieval metrics + opt-in LLM-judge.
- **5.6** [`src/router.py`](../src/router.py) — agentic routing (RAG / Web / Memory) + free DuckDuckGo search.
- **5.7** [`src/memory.py`](../src/memory.py) — conversation history + compression past a threshold.
- **5.8** [`src/storage.py`](../src/storage.py) — Supabase persistence, graceful no-op when unconfigured.
- ❓ **Data flow question:** trace a single `/api/research/stream` request through every module in order.
- ❓ **Caching:** how does `_document_signature` + `_DOCUMENT_CACHE` avoid re-embedding? What is FIFO eviction?
- ❓ **Graceful degradation:** name 3 places the app keeps working when something is missing (no embeddings → BM25; no Supabase → no-op; OCR binary missing → clear error).

---

## Part 6 — Python (with snippets straight from your code)

**6.1 Basics**
- ❓ Variables, types: `str`, `int`, `float`, `bool`, `None`.
- ❓ **Data structures:** `list` `[]`, `dict` `{}`, `set` `{}`, `tuple` `()`. When to use each? (You use all four.)
- ❓ What is an f-string? `f"{chunk['source']}_page_{chunk['page']}"`
- ❓ Slicing: `text[:700]`, `tokens[:16]`, `turns[-keep_recent:]`. ❓ What does a negative index mean?

**6.2 Control flow & functions**
- ❓ `if / elif / else`, `for`, `while`. (Your chunking `while start < len(words)`.)
- ❓ Defining functions, **default arguments**: `def retrieve_chunks(question, chunks, top_k=4)`.
- ❓ **Type hints**: `List[Dict[str, Any]]`, `Optional[str]`. ❓ Do they change runtime behavior? (No — why keep them?)
- ❓ **Return values** vs **`yield`** (generators): why does `stream_answer` use `yield`?

**6.3 Pythonic patterns in your code**
- ❓ **List/dict comprehensions:** `[c["text"] for c in chunks]`, `{**chunk, "score": s}`.
- ❓ **Dictionary unpacking:** `{**trace, "answer": answer}` — what does `**` do?
- ❓ **`.get()` with default:** `chunk.get("section", "")` — why safer than `chunk["section"]`?
- ❓ **`enumerate`** and **`zip`:** `for i, page in enumerate(pages, start=1)`, `zip(query_terms, query_terms[1:])`.
- ❓ **`sorted` with a `key`:** `ranked.sort(key=lambda c: c["rerank_score"], reverse=True)`. ❓ What is a **lambda**?
- ❓ **`Counter`** and `.most_common(8)` — what data structure is this? (Part 11.)
- ❓ **Sets for overlap:** `set(query_terms) & set(chunk_terms)` — what does `&` do?
- ❓ **`try / except`:** how your code "fails soft" (embeddings, web search, storage).
- ❓ **Context managers:** `with open(...) as f`, `with DDGS() as ddgs`. What does `with` guarantee?
- ❓ **Classes:** `class Store:`, `__init__`, `self`, methods. (See `storage.py`.)
- ❓ **Modules & imports:** `from src.retrieval import tokenize`; the `try/except ImportError` fallback — why?
- ❓ **Regular expressions:** `re.findall(r"[a-zA-Z0-9]+", text)`, `re.compile(...)`. (Part 9.4.)
- ❓ **`async def` / `await`:** why are the FastAPI upload endpoints `async`? What is concurrency vs parallelism?
- ❓ **Decorators:** `@app.get("/")`, `@app.post(...)` — what does a decorator do to the function below it?

**6.4 The environment**
- ❓ What is `pip`? `requirements.txt`? A **virtual environment** (`venv`)?
- ❓ What is `import`? Where does Python find packages? (Why did the `supabase/` folder shadow the library?)

---

## Part 7 — Backend / FastAPI / web internals

- ❓ What is **FastAPI**? What is **Uvicorn** (an ASGI server)?
- ❓ What is a **route/path operation**? How do `@app.get` / `@app.post` map URLs to functions?
- ❓ What is **Pydantic**? Why use `BaseModel` (`ChatRequest`, `ChatResponse`)? What is validation?
- ❓ What is **multipart/form-data**? Why is file upload different from JSON? (`UploadFile`, `Form(...)`.)
- ❓ What is **CORS**? Why is `allow_origins=["*"]` with credentials invalid? (You fixed this.)
- ❓ What are **Server-Sent Events (SSE)**? How does `event: chunk\ndata: {...}\n\n` work? Why SSE over WebSockets here?
- ❓ What is a **dependency** (`Depends`, `Header`)? (Your `get_optional_user`.)
- ❓ What is `StreamingResponse`? How does a generator become an HTTP stream?

---

## Part 8 — Frontend (HTML / CSS / JavaScript)

- ❓ What are **HTML** (structure), **CSS** (style), **JavaScript** (behavior)?
- ❓ What is the **DOM**? What does `document.getElementById(...)` return?
- ❓ **Events:** `addEventListener("click", ...)` — what is an event listener / callback?
- ❓ **`fetch`** and `async/await` in JS — how does the browser call your API?
- ❓ How do you **read a stream** in JS? (`response.body.getReader()`, `TextDecoder`, splitting on `\n\n`.)
- ❓ What is **`localStorage`** (your saved name)? What is **state** (`selectedFiles`, `conversationTurns`)?
- ❓ **XSS / `escapeHtml`:** why must you escape user/document text before putting it in `innerHTML`?
- ❓ **CSS layout:** what is `display: grid` / `flex`? How does the 3-column `app-shell` work?

---

## Part 9 — The math & algorithms behind retrieval

**9.1 Vectors & similarity**
- ❓ What is a **vector**? What is the **dot product**? What is a **norm** (length)?
- ❓ What is **cosine similarity**? Why "angle between vectors" = "similarity of meaning"? (`cosine_similarity`.)
- ❓ Why **normalize** embeddings? What does cosine of normalized vectors equal?

**9.2 Lexical scoring**
- ❓ What is **term frequency (TF)** and **inverse document frequency (IDF)**?
- ❓ What is **TF-IDF**? What is **BM25** and how does it improve on TF-IDF (saturation `k1`, length-norm `b`)? (`bm25_scores`.)
- ❓ What is a **stop word** and why remove them? (`STOP_WORDS`.)

**9.3 Combining scores**
- ❓ What is **min-max normalization** and why is it needed before blending BM25 + cosine? (`_min_max_normalize`.)
- ❓ Why weight semantic 0.6 / lexical 0.4? What happens at the extremes?

**9.4 Tokenization & text processing**
- ❓ What does `tokenize` do step-by-step? Why `len(token) > 2` and lowercase?
- ❓ What is a **regex character class** `[a-zA-Z0-9]+`? What does `+` mean?

---

## Part 10 — Data, databases & persistence

- ❓ What is a **database**? What is **SQL**? What are tables, rows, columns, primary keys, foreign keys?
- ❓ What is **PostgreSQL**? What is **Supabase** (managed Postgres + auth + storage)?
- ❓ What is **pgvector**? How does a `vector(1536)` column + `<=>` operator do similarity search? (`db/schema.sql`.)
- ❓ What is an **index** (e.g. `ivfflat`) and why does it speed up vector search?
- ❓ What is **Row-Level Security (RLS)**? Why does each user only see their own rows?
- ❓ What is an **RPC / SQL function** (`match_chunks`)? Why call it instead of raw SQL from Python?
- ❓ What is **persistence** vs **in-memory state**? Why does serverless lose your `_DOCUMENT_CACHE` on cold start?
- ❓ What is **stateless** vs **stateful**? Why does adding Supabase push you off pure serverless?

---

## Part 11 — Data Structures & Algorithms (as actually used here)

**11.1 Data structures in your code**
- ❓ **Hash map / dictionary** (`dict`, `Counter`, `_DOCUMENT_CACHE`) — O(1) lookup. Where and why?
- ❓ **Set** (`set`) — fast membership & intersection (`&`) for matched terms. Why a set, not a list?
- ❓ **List / array** — ordered chunks, embeddings, candidates.
- ❓ **Tuple** — fixed pairs like `(heading, body)` sections.

**11.2 Algorithms & techniques**
- ❓ **Sliding window** — your chunking moves `start` forward by `chunk_words - overlap`. What's the time complexity?
- ❓ **Sorting** — `sort(key=..., reverse=True)`; what algorithm does Python use (Timsort)? Big-O?
- ❓ **Hashing** — `hashlib.sha256` for the cache key; what is a hash function? collision? why deterministic?
- ❓ **FIFO eviction** — bounding the cache (`pop(next(iter(...)))`); what is a queue?
- ❓ **Linear scan vs index** — BM25 scans all chunks (O(n·m)); pgvector uses an index. Trade-off?
- ❓ **Normalization & weighted sum** — combining scores is just array math.

**11.3 Big-O foundations (general DSA)**
- ❓ What is **Big-O notation**? O(1), O(log n), O(n), O(n log n), O(n²) — examples from your code.
- ❓ Arrays vs linked lists; stacks vs queues; hash tables; trees; graphs (just enough to discuss trade-offs).
- ❓ Searching (linear, binary) and sorting basics; recursion.

---

## Part 12 — LLMOps, evaluation & production quality

- ❓ What is **evaluation** for RAG? Define **Hit Rate**, **MRR**, **Recall@K**, **Precision@K** (you compute all four).
- ❓ What is **Faithfulness** vs **Answer Relevance**, and why do they need an **LLM-as-judge** (and cost money)?
- ❓ Why keep cheap metrics free and judge-metrics opt-in? (Budget posture.)
- ❓ What is **observability**? What do latency / token usage / cost / fallback-rate tell you? (`/api/metrics`.)
- ❓ What is **cost estimation** from token counts? (`estimate_cost`, `PRICE_PER_TOKEN`.)
- ❓ What is **caching** as an optimization? What did the content-hash cache save?
- ❓ What is **graceful degradation / fallback** as a reliability pattern?

---

## Part 13 — Testing, Git, CI/CD, deployment

**13.1 Testing**
- ❓ What is a **unit test**? What is `pytest`? What is an **assertion**?
- ❓ Why are pure functions (retrieval, reranker, evaluation) easy to test **offline**?
- ❓ What is a **stub/mock** (your `summarizer = lambda text: ...` in memory tests)?
- ❓ What is **test coverage**? Why did print-scripts "not count" as tests?

**13.2 Git & GitHub**
- ❓ What is **version control**? What are commit, branch, push, pull, merge, remote, origin?
- ❓ What is `.gitignore` and why is `.env` in it?
- ❓ What is a **PR**? What does "8 commits ahead of origin/main" mean?
- ❓ What is a **username rename** + redirect (the Swarali603 → Swarali108 thing)?

**13.3 CI/CD & deployment**
- ❓ What is **CI** (GitHub Actions running `pytest` on push)? Why automate it?
- ❓ What is **deployment**? What is **Vercel**? What is **serverless** and its 250MB limit?
- ❓ Why might a build "succeed but the site looks old" (last-good-deploy on failure)?
- ❓ Persistent host (Render/Railway/Fly) vs serverless — when do you need each?

---

## Part 14 — Prompt & context engineering

- ❓ What is **prompt engineering**? How do your `MODES` tones change behavior with the same model?
- ❓ What is **context engineering**? (What you put in the window: chunks + web + history summary + rules.)
- ❓ Why does **rule ordering / wording** matter? (Your over-refusal bug = one literal rule.)
- ❓ What is **few-shot** vs **zero-shot**? What is **chain-of-thought**?
- ❓ How does **history compression** keep the context small while preserving meaning?

---

## Part 15 — AI safety, governance & security

- ❓ What is **grounding** and why does it matter for trust?
- ❓ What is **prompt injection**? Could a malicious uploaded doc or web page attack your prompt? How to mitigate?
- ❓ What is **PII / data privacy**? What happens to uploaded files (in-memory vs stored)?
- ❓ Why never expose the **service-role key** to the browser? What is least-privilege?
- ❓ What are **rate limits / abuse / DoS**? Why the upload-size and embed-chunk caps?
- ❓ What is **hallucination risk** and how do confidence + citations + fallback address it (AI governance)?

---

## Part 16 — Interview / viva question bank (rapid fire)

Be able to answer each in ~60 seconds:
1. "Walk me through your architecture end to end."
2. "Why hybrid retrieval instead of pure embeddings?"
3. "How does your reranker work and why add it?"
4. "How do you keep it within a $1/month budget?" (caps, caching, free routing, opt-in judge.)
5. "How do you evaluate retrieval quality?"
6. "What happens when the embedding API fails?" (BM25 fallback.)
7. "Why OpenRouter, and how do you swap models?"
8. "How does streaming work from server to browser?"
9. "How does caching avoid re-embedding the same PDF?"
10. "What's the difference between your 6 modes under the hood?" (Prompt + temperature + max_tokens.)
11. "How would you add a real vector DB?" (pgvector / `match_chunks`.)
12. "What are the security risks and how did you reduce them?"
13. "What would you improve next, and what are the trade-offs?"

---

## How to study this (a 6-week sketch)
- **Week 1:** Parts 1–3 (foundations, APIs, LLMs).
- **Week 2:** Part 6 (Python) + read your own `retrieval.py` line by line.
- **Week 3:** Part 4 + Part 9 (RAG + the math).
- **Week 4:** Parts 5, 7, 8 (your architecture, backend, frontend).
- **Week 5:** Parts 10–12 (data, DSA, LLMOps).
- **Week 6:** Parts 13–16 (testing/deploy, prompt/context, safety, mock interview).

> Want full written answers? Ask me to "expand Part N" and I'll turn any section into a
> detailed explainer with diagrams and code walkthroughs from your repo.
