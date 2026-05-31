import html
import sys
import time
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import (
    fixed_chunking,
    recursive_chunking,
    semantic_chunking,
    document_aware_chunking,
)
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import RAGPipeline

try:
    from src.external_search import search_web
except ImportError:
    search_web = None


st.set_page_config(
    page_title="AI RAG Research Assistant",
    page_icon="PDF",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #09090B;
        --panel: #0D0D10;
        --card: #18181B;
        --card-soft: #202024;
        --cyan: #00F5D4;
        --purple: #7C3AED;
        --pink: #F472B6;
        --green: #62F3A9;
        --blue: #60A5FA;
        --red: #FB7185;
        --text: #FAFAFA;
        --muted: rgba(250, 250, 250, 0.68);
        --border: rgba(250, 250, 250, 0.12);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 18% 8%, rgba(124, 58, 237, 0.16), transparent 28%),
            radial-gradient(circle at 78% 6%, rgba(0, 245, 212, 0.10), transparent 26%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 8rem;
        max-width: 1640px;
    }

    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
        background: linear-gradient(180deg, #050506 0%, #0D0D10 100%);
        border-right: 1px solid rgba(250, 250, 250, 0.14);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: 0;
    }

    .hero-title {
        margin: 0 0 0.35rem 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0;
        background: linear-gradient(90deg, #C4B5FD, var(--cyan), var(--pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.25rem;
    }

    .metric-card {
        padding: 1.05rem;
        min-height: 128px;
        background: linear-gradient(180deg, rgba(24, 24, 27, 0.88), rgba(12, 12, 15, 0.94));
        border: 1px solid var(--border);
        border-radius: 18px;
    }

    .metric-card.purple {
        border-color: rgba(124, 58, 237, 0.38);
        box-shadow: inset 0 0 30px rgba(124, 58, 237, 0.08);
    }

    .metric-card.green {
        border-color: rgba(98, 243, 169, 0.34);
        box-shadow: inset 0 0 30px rgba(98, 243, 169, 0.07);
    }

    .metric-card.blue {
        border-color: rgba(96, 165, 250, 0.34);
        box-shadow: inset 0 0 30px rgba(96, 165, 250, 0.07);
    }

    .metric-card.red {
        border-color: rgba(251, 113, 133, 0.34);
        box-shadow: inset 0 0 30px rgba(251, 113, 133, 0.07);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: 0.35rem;
    }

    .conversation-shell {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(9, 9, 11, 0.72), rgba(9, 9, 11, 0.88));
        border: 1px solid var(--border);
    }

    .message-row {
        display: flex;
        width: 100%;
        margin: 0.9rem 0;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .message-bubble {
        max-width: 76%;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        line-height: 1.58;
        font-size: 1rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
    }

    .message-bubble.user {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.55), rgba(24, 24, 27, 0.95));
        border: 1px solid rgba(196, 181, 253, 0.32);
        color: var(--text);
    }

    .message-bubble.assistant {
        background: linear-gradient(180deg, rgba(24, 24, 27, 0.95), rgba(12, 12, 15, 0.98));
        border: 1px solid rgba(0, 245, 212, 0.24);
        color: var(--text);
    }

    .empty-chat {
        color: var(--muted);
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(24, 24, 27, 0.42);
    }

    .status-panel {
        position: sticky;
        top: 1.25rem;
        overflow-wrap: anywhere;
    }

    .right-card {
        padding: 1rem;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(24, 24, 27, 0.86), rgba(10, 10, 12, 0.94));
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.25);
    }

    .status-line {
        margin-bottom: 0.7rem;
        line-height: 1.45;
        color: var(--text);
    }

    .status-label {
        color: var(--cyan);
    }

    .pipeline-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.78rem;
        margin: 0.55rem 0;
        background: rgba(24, 24, 27, 0.72);
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    .pipeline-num {
        width: 2rem;
        height: 2rem;
        display: grid;
        place-items: center;
        border-radius: 999px;
        color: var(--cyan);
        background: rgba(0, 245, 212, 0.10);
        border: 1px solid rgba(0, 245, 212, 0.25);
        font-weight: 700;
        flex: 0 0 auto;
    }

    .pipeline-title {
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.12rem;
    }

    .pipeline-subtitle {
        color: var(--muted);
        font-size: 0.84rem;
    }

    div[data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(0, 245, 212, 0.20);
    }

    .stButton > button {
        width: 100%;
        background: var(--card);
        color: var(--text);
        border: 1px solid rgba(0, 245, 212, 0.36);
        border-radius: 12px;
        padding: 0.65rem 1rem;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: var(--cyan);
        box-shadow: 0 0 18px rgba(0, 245, 212, 0.18);
    }

    [data-testid="stFileUploader"] section {
        background: rgba(9, 9, 11, 0.92);
        border: 1px dashed rgba(196, 181, 253, 0.45);
        border-radius: 16px;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: var(--purple);
        border: 2px solid var(--bg);
    }

    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--purple));
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        left: 530px;
        right: 540px;
        bottom: 18px;
        z-index: 999;
        background: rgba(9, 9, 11, 0.96);
        padding: 0.45rem;
        border: 1px solid rgba(196, 181, 253, 0.24);
        border-radius: 18px;
        box-shadow: 0 0 30px rgba(124, 58, 237, 0.18);
    }

    div[data-testid="stChatInput"] textarea {
        background: var(--card);
        color: var(--text);
    }

    @media (max-width: 1300px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        div[data-testid="stChatInput"] {
            left: 390px;
            right: 40px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    defaults = {
        "rag_pipeline": None,
        "chunks": [],
        "processed_file_name": None,
        "processed_document_count": 0,
        "messages": [],
        "last_result": None,
        "last_chunking_method": None,
        "average_chunk_size": 0,
        "last_response_time": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_average_chunk_size(chunks):
    if not chunks:
        return 0

    total_words = sum(len(chunk["text"].split()) for chunk in chunks)
    return round(total_words / len(chunks))


def average_similarity_score(result):
    citations = result.get("citations", [])

    if not citations:
        return 0

    return sum(item["score"] for item in citations) / len(citations)


def retrieval_accuracy_percent(result):
    avg_score = average_similarity_score(result)
    return max(0, min(100, round(avg_score * 100)))


def answer_confidence_percent(result):
    avg_score = average_similarity_score(result)
    external_results = result.get("external_results", [])
    confidence = avg_score

    if external_results:
        confidence = min(1.0, confidence + 0.08)

    return max(0, min(100, round(confidence * 100)))


def should_show_low_confidence_warning(result):
    answer = result.get("answer", "").lower()
    avg_score = average_similarity_score(result)

    not_found_phrases = [
        "i do not know",
        "not found in the uploaded document",
        "not found in the uploaded documents",
        "not mentioned in the uploaded document",
        "not mentioned in the uploaded documents",
        "not provided in the uploaded document",
        "not provided in the uploaded documents",
        "cannot answer based on the uploaded document",
        "cannot answer based on the uploaded documents",
        "answer is not present",
        "not present in the context",
        "not available in the provided context",
    ]

    model_says_not_found = any(phrase in answer for phrase in not_found_phrases)

    return model_says_not_found and avg_score < 0.45


def estimate_source_mix(result):
    external_results = result.get("external_results", [])
    citations = result.get("citations", [])

    if not external_results:
        return 100, 0

    if not citations:
        return 0, 100

    average_doc_score = average_similarity_score(result)
    web_weight = min(len(external_results) * 0.22, 0.66)
    doc_weight = max(average_doc_score, 0.05)
    total_weight = doc_weight + web_weight

    document_percent = round((doc_weight / total_weight) * 100)
    web_percent = 100 - document_percent

    return document_percent, web_percent


def render_message(role, content):
    safe_content = html.escape(content)

    st.markdown(
        f"""
        <div class="message-row {role}">
            <div class="message-bubble {role}">
                {safe_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards():
    document_count = st.session_state.processed_document_count or 0
    chunk_count = len(st.session_state.chunks)
    avg_similarity = 0

    if st.session_state.last_result:
        avg_similarity = average_similarity_score(st.session_state.last_result)

    response_time = st.session_state.last_response_time

    response_display = "0.0s"
    response_note = "Waiting"

    if response_time is not None:
        response_display = f"{response_time:.1f}s"
        response_note = "Ready"

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card purple">
                <div class="metric-label">Documents</div>
                <div class="metric-value" style="color:#C4B5FD;">{document_count}</div>
                <div class="metric-note">Uploaded</div>
            </div>
            <div class="metric-card green">
                <div class="metric-label">Chunks</div>
                <div class="metric-value" style="color:#62F3A9;">{chunk_count}</div>
                <div class="metric-note">Created</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-label">Avg. Similarity</div>
                <div class="metric-value" style="color:#60A5FA;">{avg_similarity:.2f}</div>
                <div class="metric-note">Retrieval match</div>
            </div>
            <div class="metric-card red">
                <div class="metric-label">Response Time</div>
                <div class="metric-value" style="color:#FB7185;">{response_display}</div>
                <div class="metric-note">{response_note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_item(number, title, status):
    st.markdown(
        f"""
        <div class="pipeline-item">
            <div class="pipeline-num">{number}</div>
            <div>
                <div class="pipeline-title">{html.escape(title)}</div>
                <div class="pipeline-subtitle">{html.escape(status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


initialize_session_state()


with st.sidebar:
    st.header("Document Settings")

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunking_method = st.selectbox(
        "Chunking Method",
        ["Recursive", "Fixed", "Semantic", "Document-aware"],
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values are predictable. Higher values are more creative.",
    )

    use_external_search = st.checkbox(
        "Use external web search",
        value=False,
    )

    process_button = st.button("Process Document")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.last_response_time = None
        st.rerun()


center_col, status_col = st.columns([3.2, 1.35], gap="large")


with center_col:
    st.markdown(
        '<h1 class="hero-title">AI RAG Research Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Upload a PDF and ask source-grounded questions with citations.</div>',
        unsafe_allow_html=True,
    )

    render_metric_cards()

    process_status = st.empty()

    st.markdown('<div class="conversation-shell">', unsafe_allow_html=True)
    st.subheader("Conversation")

    chat_window = st.container(height=560)

    with chat_window:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="empty-chat">
                    Hi. Ask me anything about your uploaded documents. I will answer with citations from your PDF.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state.messages:
            render_message(message["role"], message["content"])

        if st.session_state.last_result:
            result = st.session_state.last_result

            with st.expander("Answer Details", expanded=False):
                if should_show_low_confidence_warning(result):
                    st.warning("Answer may not exist in uploaded documents.")

                if "standalone_question" in result:
                    st.write("Standalone Retrieval Question")
                    st.write(result["standalone_question"])

                doc_percent, web_percent = estimate_source_mix(result)
                st.write("Estimated Source Mix")
                st.write(f"Uploaded document: {doc_percent}%")
                st.write(f"External web search: {web_percent}%")

                st.write("Retrieval Accuracy")
                st.write(f"{retrieval_accuracy_percent(result)}%")

                st.write("Answer Confidence")
                st.write(f"{answer_confidence_percent(result)}%")

                st.caption(
                    "Scores are estimates based on retrieved chunk similarity and optional web context."
                )

                st.write("Citations")
                for citation in result["citations"]:
                    st.write(
                        f"{citation.get('source', 'uploaded_document.pdf')} | "
                        f"Page {citation['page']} | "
                        f"Chunk: {citation['chunk_id']} | "
                        f"Score: {citation['score']:.3f}"
                    )

                if result.get("external_results"):
                    st.write("External Sources")
                    for index, item in enumerate(result["external_results"], start=1):
                        st.write(f"Web Source {index}: [{item['title']}]({item['url']})")

                st.write("Retrieved Chunks")
                for index, chunk in enumerate(result["retrieved_chunks"], start=1):
                    st.markdown(
                        f"**Source {index} - "
                        f"{chunk.get('source', 'uploaded_document.pdf')} - "
                        f"Page {chunk['page']}**"
                    )
                    st.write(chunk["text"])
                    st.divider()

    st.markdown("</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your document...")


with status_col:
    st.markdown('<div class="status-panel">', unsafe_allow_html=True)

    st.markdown('<div class="right-card">', unsafe_allow_html=True)
    st.subheader("Project Status")

    if st.session_state.processed_file_name:
        st.markdown(
            f'<div class="status-line"><span class="status-label">Document:</span> {st.session_state.processed_file_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-line"><span class="status-label">Chunks:</span> {len(st.session_state.chunks)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-line"><span class="status-label">Messages:</span> {len(st.session_state.messages)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-line"><span class="status-label">Chunking:</span> {st.session_state.last_chunking_method}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-line"><span class="status-label">Average Chunk Size:</span> {st.session_state.average_chunk_size} words</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write("No document processed yet.")

    if st.session_state.last_result:
        result = st.session_state.last_result
        doc_percent, web_percent = estimate_source_mix(result)
        retrieval_percent = retrieval_accuracy_percent(result)
        confidence_percent = answer_confidence_percent(result)

        st.write("Source Mix")
        st.progress(doc_percent / 100)
        st.write(f"Uploaded document estimate: {doc_percent}%")
        st.write(f"External web estimate: {web_percent}%")

        st.write("Retrieval Accuracy")
        st.progress(retrieval_percent / 100)
        st.write(f"{retrieval_percent}%")

        st.write("Answer Confidence")
        st.progress(confidence_percent / 100)
        st.write(f"{confidence_percent}%")

        if should_show_low_confidence_warning(result):
            st.warning("Answer may not exist in uploaded documents.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="right-card">', unsafe_allow_html=True)
    st.subheader("Detailed Pipeline")

    pipeline_status = "Completed" if st.session_state.rag_pipeline else "Waiting"
    memory_status = "Active" if st.session_state.messages else "Waiting"
    external_status = "Enabled" if use_external_search else "Disabled"

    render_pipeline_item("1", "PDF Upload", pipeline_status)
    render_pipeline_item("2", "Text Extraction", pipeline_status)
    render_pipeline_item("3", "Chunking", pipeline_status)
    render_pipeline_item("4", "Embeddings", pipeline_status)
    render_pipeline_item("5", "FAISS Retrieval", pipeline_status)
    render_pipeline_item("6", "Conversation Memory", memory_status)
    render_pipeline_item("7", "External Search", external_status)
    render_pipeline_item("8", "Gemini Answer", pipeline_status)
    render_pipeline_item("9", "Citations", pipeline_status)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if process_button:
    if not uploaded_files:
        process_status.error("Please upload at least one PDF first.")
    else:
        all_pages = []

        process_status.info("Reading PDF files...")
        for uploaded_file in uploaded_files:
            pages = load_pdf(uploaded_file)

            for page in pages:
                page["source"] = uploaded_file.name

            all_pages.extend(pages)

        if not all_pages:
            process_status.error("No text could be extracted from the uploaded PDF files.")
        else:
            process_status.info("Loading embedding model...")
            embedding_model = EmbeddingModel()

            process_status.info("Chunking document...")
            if chunking_method == "Fixed":
                chunks = fixed_chunking(all_pages)
                actual_chunking_method = "Fixed"
            elif chunking_method == "Semantic":
                chunks = semantic_chunking(all_pages, embedding_model)
                actual_chunking_method = "Semantic"
            elif chunking_method == "Document-aware":
                chunks = document_aware_chunking(all_pages)
                actual_chunking_method = "Document-aware"
            else:
                chunks = recursive_chunking(all_pages)
                actual_chunking_method = "Recursive"

            process_status.info("Generating embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = embedding_model.embed_texts(texts)

            process_status.info("Building FAISS vector store...")
            vector_store = VectorStore(chunk_embeddings.shape[1])
            vector_store.add_embeddings(chunk_embeddings, chunks)

            process_status.info("Connecting Gemini...")
            rag_pipeline = RAGPipeline(embedding_model, vector_store)

            file_names = ", ".join([file.name for file in uploaded_files])

            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.chunks = chunks
            st.session_state.processed_file_name = file_names
            st.session_state.processed_document_count = len(uploaded_files)
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.last_chunking_method = actual_chunking_method
            st.session_state.average_chunk_size = get_average_chunk_size(chunks)
            st.session_state.last_response_time = None

            process_status.success(
                f"Processed {file_names}. Created {len(chunks)} chunks using "
                f"{actual_chunking_method.lower()} chunking."
            )


if question:
    if st.session_state.rag_pipeline is None:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        external_results = []
        start_time = time.perf_counter()

        with st.spinner("Retrieving context and generating answer..."):
            if use_external_search:
                if search_web is None:
                    st.warning("External search is not available. Install duckduckgo-search first.")
                else:
                    external_results = search_web(question, max_results=3)

            result = st.session_state.rag_pipeline.ask(
                question=question,
                top_k=top_k,
                temperature=temperature,
                chat_history=st.session_state.messages[:-1],
                external_results=external_results,
            )

        elapsed_time = time.perf_counter() - start_time

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
        })

        st.session_state.last_result = result
        st.session_state.last_response_time = elapsed_time
        st.rerun()

        