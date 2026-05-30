import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import fixed_chunking, recursive_chunking
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import RAGPipeline


st.set_page_config(
    page_title="AI RAG Research Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("AI RAG Research Assistant")
st.write("Upload a PDF and ask source-grounded questions with citations.")

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None


with st.sidebar:
    st.header("Document Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    chunking_method = st.selectbox(
        "Chunking method",
        ["Recursive", "Fixed"]
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=3
    )

    process_button = st.button("Process Document")


if process_button:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Reading PDF..."):
            pages = load_pdf(uploaded_file)

        if not pages:
            st.error("No text could be extracted from this PDF. Try a text-based PDF.")
        else:
            with st.spinner("Chunking document..."):
                if chunking_method == "Fixed":
                    chunks = fixed_chunking(pages)
                else:
                    chunks = recursive_chunking(pages)

            with st.spinner("Generating embeddings..."):
                embedding_model = EmbeddingModel()
                texts = [chunk["text"] for chunk in chunks]
                chunk_embeddings = embedding_model.embed_texts(texts)

            with st.spinner("Building FAISS vector store..."):
                vector_store = VectorStore(chunk_embeddings.shape[1])
                vector_store.add_embeddings(chunk_embeddings, chunks)

            with st.spinner("Connecting Gemini..."):
                rag_pipeline = RAGPipeline(embedding_model, vector_store)

            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.chunks = chunks
            st.session_state.processed_file_name = uploaded_file.name

            st.success(f"Processed {uploaded_file.name}")
            st.info(f"Created {len(chunks)} chunks using {chunking_method.lower()} chunking.")


left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Ask a Question")

    question = st.text_input(
        "Question",
        placeholder="Example: What is artificial intelligence?"
    )

    ask_button = st.button("Ask")

    if ask_button:
        if st.session_state.rag_pipeline is None:
            st.error("Please upload and process a PDF first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                result = st.session_state.rag_pipeline.ask(
                    question=question,
                    top_k=top_k
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Citations")
            for citation in result["citations"]:
                st.write(
                    f"Page {citation['page']} | "
                    f"Chunk: {citation['chunk_id']} | "
                    f"Score: {citation['score']:.3f}"
                )

            with st.expander("View Retrieved Chunks"):
                for index, chunk in enumerate(result["retrieved_chunks"], start=1):
                    st.markdown(f"**Source {index} — Page {chunk['page']}**")
                    st.write(chunk["text"])
                    st.divider()


with right_col:
    st.subheader("Project Status")

    if st.session_state.processed_file_name:
        st.write(f"Document: {st.session_state.processed_file_name}")
        st.write(f"Chunks: {len(st.session_state.chunks)}")
    else:
        st.write("No document processed yet.")

    st.subheader("Pipeline")
    st.write("PDF Upload")
    st.write("Text Extraction")
    st.write("Chunking")
    st.write("Embeddings")
    st.write("FAISS Retrieval")
    st.write("Gemini Answer")
    st.write("Citations")
    