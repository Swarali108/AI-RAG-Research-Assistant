import os
from dotenv import load_dotenv
from google import genai


load_dotenv()


class RAGPipeline:
    def __init__(self, embedding_model, vector_store, model_name="gemini-2.5-flash"):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

        self.client = genai.Client(api_key=api_key)

    def build_prompt(self, question, retrieved_chunks):
        context = ""

        for index, chunk in enumerate(retrieved_chunks, start=1):
            context += f"""
Source {index}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Content:
{chunk["text"]}
"""

        prompt = f"""
You are an AI Research Assistant.

Answer the user's question using only the provided context.

Rules:
- If the answer is not present in the context, say: "I do not know based on the uploaded document."
- Do not make up facts.
- Include page citations in the answer using this format: [Page X].
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
        return prompt

    def ask(self, question, top_k=3):
        query_embedding = self.embedding_model.embed_query(question)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)

        prompt = self.build_prompt(question, retrieved_chunks)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        citations = [
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"]
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": response.text,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks
        }
    