import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel


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

    return genai.Client(api_key=api_key)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "AI RAG Research Assistant API is running on Vercel",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_message = request.message.strip()

    if not user_message:
        return ChatResponse(
            answer="Please enter a question.",
            follow_up_questions=[],
        )

    history_text = ""

    for message in request.history[-6:]:
        role = message.get("role", "user")
        content = message.get("content", "")
        history_text += f"{role.upper()}: {content}\n"

    prompt = f"""
You are an AI Research Assistant.

Answer clearly, accurately, and professionally.
If the user asks for explanation, make it simple and structured.
If you do not know something, say so instead of inventing facts.

Conversation history:
{history_text}

User question:
{user_message}

Answer:
"""

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        answer = response.text or "I could not generate an answer."

    except Exception as error:
        answer = f"Gemini API error: {str(error)}"

    follow_up_questions = [
        "Can you explain this in simpler terms?",
        "Can you give an example?",
        "Can you summarize the key points?",
    ]

    return ChatResponse(
        answer=answer,
        follow_up_questions=follow_up_questions,
    )
       