"""Supabase persistence layer (#1, #9, #10-persisted).

Provides persistent user workspaces (profiles, saved documents, saved chats),
pgvector-backed embedding storage, and persisted observability — all behind a
single ``Store`` that DEGRADES GRACEFULLY: if SUPABASE_URL / SUPABASE_KEY are
not set (or the ``supabase`` package isn't installed), ``Store.enabled`` is
False and every method becomes a safe no-op. The app then runs exactly as it
does today (stateless, in-memory), so nothing breaks before you provision a
project.

Activation:
1. Create a Supabase project, enable the ``vector`` extension, and run
   ``db/schema.sql``.
2. Set SUPABASE_URL and SUPABASE_KEY (service-role key for the backend) in your
   environment. Optionally SUPABASE_JWT_SECRET to verify user access tokens.
"""

import os
from typing import Any, Dict, List, Optional


class Store:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "").strip()
        self.key = os.getenv("SUPABASE_KEY", "").strip()
        self._client = None
        self.enabled = False

        if self.url and self.key:
            try:
                from supabase import create_client

                self._client = create_client(self.url, self.key)
                self.enabled = True
            except Exception:
                self._client = None
                self.enabled = False

    # --- auth -------------------------------------------------------------
    def user_from_token(self, access_token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve a Supabase access token to a user, or None (anonymous)."""
        if not self.enabled or not access_token:
            return None
        try:
            response = self._client.auth.get_user(access_token)
            user = getattr(response, "user", None)
            if user is None:
                return None
            return {"id": user.id, "email": getattr(user, "email", None)}
        except Exception:
            return None

    # --- profiles ---------------------------------------------------------
    def upsert_profile(self, user_id: str, data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self._client.table("profiles").upsert({"id": user_id, **data}).execute()
            return True
        except Exception:
            return False

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            res = self._client.table("profiles").select("*").eq("id", user_id).limit(1).execute()
            return (res.data or [None])[0]
        except Exception:
            return None

    # --- documents --------------------------------------------------------
    def save_document(self, user_id: str, name: str, signature: str, meta: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self._client.table("documents").upsert(
                {"user_id": user_id, "name": name, "signature": signature, "meta": meta}
            ).execute()
            return True
        except Exception:
            return False

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            res = (
                self._client.table("documents")
                .select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            )
            return res.data or []
        except Exception:
            return []

    # --- chats ------------------------------------------------------------
    def save_chat(self, user_id: str, title: str, turns: List[Dict[str, str]]) -> bool:
        if not self.enabled:
            return False
        try:
            self._client.table("chats").insert(
                {"user_id": user_id, "title": title, "turns": turns}
            ).execute()
            return True
        except Exception:
            return False

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            res = (
                self._client.table("chats")
                .select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            )
            return res.data or []
        except Exception:
            return []

    # --- pgvector embeddings ---------------------------------------------
    def save_embeddings(self, document_signature: str, chunks: List[Dict[str, Any]],
                        embeddings: List[List[float]]) -> bool:
        if not self.enabled or not embeddings:
            return False
        try:
            rows = [
                {
                    "document_signature": document_signature,
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["text"],
                    "page": chunk.get("page"),
                    "embedding": embedding,
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]
            self._client.table("chunk_embeddings").insert(rows).execute()
            return True
        except Exception:
            return False

    def match_chunks(self, document_signature: str, query_embedding: List[float], k: int = 12):
        """Vector similarity search via the match_chunks SQL function (pgvector)."""
        if not self.enabled:
            return []
        try:
            res = self._client.rpc(
                "match_chunks",
                {"sig": document_signature, "query_embedding": query_embedding, "match_count": k},
            ).execute()
            return res.data or []
        except Exception:
            return []

    # --- observability ----------------------------------------------------
    def log_metrics(self, record: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self._client.table("request_metrics").insert(record).execute()
            return True
        except Exception:
            return False


# Module-level singleton used by the app.
store = Store()
