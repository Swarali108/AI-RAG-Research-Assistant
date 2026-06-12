from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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


class ChatResponse(BaseModel):
    answer: str
    follow_up_questions: List[str] = []


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

    # TODO: Connect your existing RAG / AI logic here.
    # For now, this keeps the Vercel deployment working.
    answer = f"Received your question: {user_message}"

    follow_up_questions = [
        "Can you explain this in more detail?",
        "What sources support this answer?",
        "Can you summarize the key points?",
    ]

    return ChatResponse(
        answer=answer,
        follow_up_questions=follow_up_questions,
    )
              