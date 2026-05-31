import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class RAGPipeline:
    def __init__(self, embedding_model, vector_store, model_name="gemini-2.5-flash"):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

        self.client = genai.Client(api_key=api_key)

    def format_chat_history(self, chat_history):
        formatted = ""

        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted += f"{role.upper()}: {content}\n"

        return formatted

    def rewrite_question(self, question, chat_history):
        if not chat_history:
            return question

        history_text = self.format_chat_history(chat_history[-6:])

        prompt = f"""
Rewrite the latest user question as a standalone question.

Use the conversation history to resolve references like it, this, that, they, he, she.

Conversation history:
{history_text}

Latest question:
{question}

Standalone question:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )

        return response.text.strip()

    def build_prompt(
        self,
        question,
        standalone_question,
        retrieved_chunks,
        chat_history=None,
        external_results=None,
    ):
        document_context = ""

        for index, chunk in enumerate(retrieved_chunks, start=1):
            document_context += f"""
Document Source {index}
File: {chunk.get("source", "uploaded_document.pdf")}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Content:
{chunk["text"]}
"""

        web_context = ""

        if external_results:
            for index, result in enumerate(external_results, start=1):
                web_context += f"""
Web Source {index}
Title: {result.get("title", "Untitled")}
URL: {result.get("url", "")}
Snippet:
{result.get("snippet", "")}
"""

        history_text = self.format_chat_history(chat_history[-6:]) if chat_history else ""

        prompt = f"""
You are an AI Research Assistant.

Answer the user using:
1. Uploaded document context first.
2. External web context only when provided.
3. Conversation history only to understand references.

Rules:
- Prefer uploaded document evidence over web results.
- If the answer is not supported by the uploaded document or web context, say you do not know.
- Cite uploaded documents as [File name, Page X].
- Cite web results as [Web Source X].
- Do not invent facts.

Conversation History:
{history_text}

Uploaded Document Context:
{document_context}

External Web Context:
{web_context}

Original User Question:
{question}

Standalone Retrieval Question:
{standalone_question}

Answer:
"""
        return prompt

    def ask(
        self,
        question,
        top_k=3,
        temperature=0.2,
        chat_history=None,
        external_results=None,
    ):
        chat_history = chat_history or []
        external_results = external_results or []

        standalone_question = self.rewrite_question(question, chat_history)

        query_embedding = self.embedding_model.embed_query(standalone_question)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)

        prompt = self.build_prompt(
            question=question,
            standalone_question=standalone_question,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            external_results=external_results,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )

        citations = [
            {
                "source": chunk.get("source", "uploaded_document.pdf"),
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": response.text,
            "standalone_question": standalone_question,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "external_results": external_results,
        }
    