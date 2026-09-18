import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


RESEARCH_PROMPT = """
You are an AI Research Assistant.

Your job is to provide accurate, professional, clear, and concise answers.

Rules:
- Use professional language.
- Use technical terminology when appropriate.
- Be factual and evidence-based.
- Explain concepts clearly.
- Use the provided context as the primary source.
- Never invent information.
- If information is missing, say so.
- Keep answers structured and easy to read.
- Maintain a neutral and professional tone.
- Always include citations when document sources are used.

Always prioritize accuracy over creativity.
"""


GENZ_PROMPT = """
You are a highly intelligent best friend.

Your job is to explain things like a smart, slightly chaotic friend who genuinely wants someone to understand something.

You are funny, relatable, sarcastic when appropriate, and occasionally dramatic.

You may use Gen-Z expressions such as:
- bestie
- fr
- ngl
- srsly
- bro
- girl
- damnnnn
- iykyk
- respectfully
- that's wild
- absolutely cooked
- not gonna lie
- this is giving
- lowkey
- highkey

IMPORTANT:
- Do NOT use slang in every sentence.
- Do NOT sound like a TikTok comment section.
- Do NOT sacrifice accuracy.
- Do NOT make up information.
- Do NOT use cringe phrases repeatedly.
- Keep explanations helpful first and funny second.
- Always include citations when document sources are used.

The goal is: smart friend explaining difficult concepts over coffee.

Use analogies, relatable examples, light humor, occasional sarcasm, and occasional emojis.

Your personality should feel like:
70% smart friend
20% comedian
10% chaos

Never become a meme generator.
The user came to learn something. Help them understand it.
"""


PROFESSIONAL_FORMATTING_TRIGGERS = [
    "write a report",
    "draft a report",
    "create a report",
    "write an email",
    "draft an email",
    "create documentation",
    "write documentation",
    "make documentation",
    "professional summary",
    "formal summary",
    "prepare a report",
]


class RAGPipeline:
    def __init__(self, embedding_model, vector_store, model_name="gemini-2.5-flash-lite"):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.model_name = model_name
        self.last_debug = {}

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it in Streamlit Secrets.")

        self.client = genai.Client(api_key=str(api_key).strip())

    def get_system_prompt(self, question, answer_mode):
        question_lower = question.lower()
        should_force_research = any(
            trigger in question_lower for trigger in PROFESSIONAL_FORMATTING_TRIGGERS
        )

        if should_force_research:
            return RESEARCH_PROMPT, "Research Mode"

        if answer_mode == "✨ Gen-Z Mode":
            return GENZ_PROMPT, "Gen-Z Mode"

        return RESEARCH_PROMPT, "Research Mode"

    def format_chat_history(self, chat_history):
        formatted = ""

        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted += f"{role.upper()}: {content}\n"

        return formatted

    def route_question(self, question, chat_history=None, use_external_search=False):
        q = question.lower()
        chat_history = chat_history or []

        current_event_words = [
            "latest", "today", "current", "recent", "news", "2026", "now"
        ]

        memory_words = [
            "it", "this", "that", "they", "those", "same", "above", "previous"
        ]

        if use_external_search or any(word in q for word in current_event_words):
            return "Web Search"

        if chat_history and any(word in q.split() for word in memory_words):
            return "Conversation Memory"

        return "Document RAG"

    def rewrite_question(self, question, chat_history):
        if not chat_history:
            return question

        history_text = self.format_chat_history(chat_history[-6:])

        prompt = f"""
Rewrite the latest user question as a standalone retrieval question.

Important:
- Preserve the user's intent.
- Resolve references like it, this, that, they.
- Do not convert follow-up questions into generic definitions.
- If the user asks "where do we use it?", preserve that as a usage question.
- If the user asks "examples?", preserve that as an examples question.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone retrieval question:
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text.strip()
        except Exception:
            return question

    def build_prompt(
        self,
        question,
        standalone_question,
        retrieved_chunks,
        chat_history=None,
        external_results=None,
        answer_mode="🧠 Research Mode",
    ):
        system_prompt, effective_mode = self.get_system_prompt(question, answer_mode)

        document_context = ""

        for index, chunk in enumerate(retrieved_chunks, start=1):
            document_context += f"""
Document Source {index}
File: {chunk.get("source", "uploaded_document.pdf")}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Similarity Score: {chunk["score"]:.3f}
Content:
{chunk["text"]}
"""

        web_context = ""

        if external_results:
            for index, result in enumerate(external_results[:3], start=1):
                title = str(result.get("title", "Untitled"))[:140]
                url = str(result.get("url", ""))[:260]
                snippet = str(result.get("snippet", ""))[:600]

                web_context += f"""
Web Source {index}
Title: {title}
URL: {url}
Snippet:
{snippet}
"""

        history_text = self.format_chat_history(chat_history[-6:]) if chat_history else ""

        final_prompt = f"""
{system_prompt}

You are answering inside an AI RAG Research Assistant.

Mode Selected:
{effective_mode}

Source Rules:
- Use uploaded document context as the primary source.
- Use external web context only when provided.
- Use conversation history only to understand references and avoid repetition.
- Always keep citations in the answer when document or web context is used.
- Cite uploaded documents as [File name, Page X].
- Cite web results as [Web Source X].
- Never invent sources.
- If the answer is not supported by the context, say you do not know based on the available sources.

Task Rules:
- Answer the latest user question directly.
- Do not repeat the same definition if it was already answered.
- If the latest question asks "how", explain practical steps.
- If the latest question asks "where", explain use cases or situations.
- If the latest question asks for examples, give concrete examples.

Conversation History:
{history_text}

Context:
{document_context}

External Web Context:
{web_context}

Question:
{question}

Standalone Retrieval Question:
{standalone_question}

Answer:
"""

        return final_prompt, effective_mode

    def _prepare_rag(
        self,
        question,
        top_k=3,
        chat_history=None,
        external_results=None,
        answer_mode="🧠 Research Mode",
    ):
        chat_history = chat_history or []
        external_results = external_results or []

        standalone_question = self.rewrite_question(question, chat_history)
        query_embedding = self.embedding_model.embed_query(standalone_question)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)

        citations = [
            {
                "source": chunk.get("source", "uploaded_document.pdf"),
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ]

        prompt, effective_mode = self.build_prompt(
            question=question,
            standalone_question=standalone_question,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            external_results=external_results,
            answer_mode=answer_mode,
        )

        self.last_debug = {
            "question": question,
            "standalone_question": standalone_question,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations,
            "prompt": prompt,
            "external_results": external_results,
            "answer_mode": answer_mode,
            "effective_mode": effective_mode,
        }

        return standalone_question, retrieved_chunks, citations, prompt

    def stream_answer(
        self,
        question,
        top_k=3,
        temperature=0.2,
        chat_history=None,
        external_results=None,
        answer_mode="🧠 Research Mode",
    ):
        self._prepare_rag(
            question=question,
            top_k=top_k,
            chat_history=chat_history,
            external_results=external_results,
            answer_mode=answer_mode,
        )

        prompt = self.last_debug["prompt"]

        try:
            stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as error:
            error_text = str(error)
            self.last_debug["error"] = error_text

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                yield "Gemini quota or rate limit reached. Wait a bit or use another Gemini API key."
            elif "API_KEY_INVALID" in error_text or "403" in error_text:
                yield "Gemini API key issue. Check Streamlit Secrets and use a fresh unrestricted Gemini API key."
            elif "INVALID_ARGUMENT" in error_text or "400" in error_text:
                yield "Gemini rejected the request. Try web search off, fewer retrieved chunks, or a shorter question."
            else:
                yield "Gemini API error while generating the answer. Try turning external web search off or checking Streamlit logs."

    def ask(
        self,
        question,
        top_k=3,
        temperature=0.2,
        chat_history=None,
        external_results=None,
        answer_mode="🧠 Research Mode",
    ):
        answer_parts = []

        for part in self.stream_answer(
            question=question,
            top_k=top_k,
            temperature=temperature,
            chat_history=chat_history,
            external_results=external_results,
            answer_mode=answer_mode,
        ):
            answer_parts.append(part)

        answer = "".join(answer_parts)
        debug = self.last_debug

        return {
            "answer": answer,
            "standalone_question": debug.get("standalone_question", question),
            "citations": debug.get("citations", []),
            "retrieved_chunks": debug.get("retrieved_chunks", []),
            "external_results": debug.get("external_results", []),
            "prompt": debug.get("prompt", ""),
            "error": debug.get("error"),
            "answer_mode": debug.get("answer_mode", answer_mode),
            "effective_mode": debug.get("effective_mode", answer_mode),
        }
      