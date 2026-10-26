"""Postgres persistence layer (#1, #9, #10-persisted), hosted on Neon.

Provides persistent user workspaces (profiles, saved documents, saved chats),
pgvector-backed embedding storage, and persisted observability — all behind a
single ``Store`` that DEGRADES GRACEFULLY: if DATABASE_URL is not set (or the
``psycopg`` package isn't installed), ``Store.enabled`` is False and every
method becomes a safe no-op. The app then runs exactly as it does today
(stateless, in-memory), so nothing breaks before you provision a database.

Activation:
1. Create a Neon project and run ``db/schema.sql`` in its SQL editor.
2. Set DATABASE_URL to the project's connection string (use the pooled one).

Auth is not wired yet: ``user_from_token`` always returns None, so workspace
endpoints answer 401 until a provider (e.g. Clerk) is added.
"""

import os
from typing import Any, Dict, List, Optional


def _vector_literal(values: List[float]) -> str:
    """pgvector's text form, so no pgvector Python adapter is needed."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class Store:
    def __init__(self):
        self.dsn = os.getenv("DATABASE_URL", "").strip()
        self._psycopg = None
        self._dict_row = None
        self._json = None
        self.enabled = False

        if self.dsn:
            try:
                import psycopg
                from psycopg.rows import dict_row
                from psycopg.types.json import Jsonb

                self._psycopg = psycopg
                self._dict_row = dict_row
                self._json = Jsonb
                self.enabled = True
            except Exception:
                self.enabled = False

    def _connect(self):
        # One short-lived connection per call; Neon's pooled endpoint absorbs the cost.
        # connect_timeout keeps an unreachable database from stalling a request.
        return self._psycopg.connect(
            self.dsn, autocommit=True, row_factory=self._dict_row, connect_timeout=5
        )

    def _execute(self, sql: str, params: Any = None) -> bool:
        with self._connect() as conn:
            conn.execute(sql, params)
        return True

    def _fetch(self, sql: str, params: Any = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    # --- auth -------------------------------------------------------------
    def user_from_token(self, access_token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve an access token to a user, or None (anonymous).

        No auth provider is configured yet, so every request is anonymous.
        """
        return None

    # --- profiles ---------------------------------------------------------
    def upsert_profile(self, user_id: str, data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            return self._execute(
                """
                insert into profiles (id, display_name) values (%s, %s)
                on conflict (id) do update set display_name = excluded.display_name
                """,
                (user_id, data.get("display_name")),
            )
        except Exception:
            return False

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            rows = self._fetch("select * from profiles where id = %s limit 1", (user_id,))
            return (rows or [None])[0]
        except Exception:
            return None

    # --- documents --------------------------------------------------------
    def save_document(self, user_id: str, name: str, signature: str, meta: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            return self._execute(
                """
                insert into documents (user_id, name, signature, meta) values (%s, %s, %s, %s)
                on conflict (user_id, signature)
                do update set name = excluded.name, meta = excluded.meta
                """,
                (user_id, name, signature, self._json(meta)),
            )
        except Exception:
            return False

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return self._fetch(
                "select * from documents where user_id = %s order by created_at desc", (user_id,)
            )
        except Exception:
            return []

    # --- chats ------------------------------------------------------------
    def save_chat(self, user_id: str, title: str, turns: List[Dict[str, str]]) -> bool:
        if not self.enabled:
            return False
        try:
            return self._execute(
                "insert into chats (user_id, title, turns) values (%s, %s, %s)",
                (user_id, title, self._json(turns)),
            )
        except Exception:
            return False

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return self._fetch(
                "select * from chats where user_id = %s order by created_at desc", (user_id,)
            )
        except Exception:
            return []

    # --- pgvector embeddings ---------------------------------------------
    def save_embeddings(self, document_signature: str, chunks: List[Dict[str, Any]],
                        embeddings: List[List[float]]) -> bool:
        if not self.enabled or not embeddings:
            return False
        try:
            rows = [
                (
                    document_signature,
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk.get("page"),
                    _vector_literal(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        insert into chunk_embeddings
                          (document_signature, chunk_id, content, page, embedding)
                        values (%s, %s, %s, %s, %s::vector)
                        """,
                        rows,
                    )
            return True
        except Exception:
            return False

    def match_chunks(self, document_signature: str, query_embedding: List[float], k: int = 12):
        """Vector similarity search via the match_chunks SQL function (pgvector)."""
        if not self.enabled:
            return []
        try:
            return self._fetch(
                "select * from match_chunks(%s, %s::vector, %s)",
                (document_signature, _vector_literal(query_embedding), k),
            )
        except Exception:
            return []

    # --- observability ----------------------------------------------------
    def log_metrics(self, record: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            columns = list(record)
            return self._execute(
                f"insert into request_metrics ({', '.join(columns)}) "
                f"values ({', '.join(['%s'] * len(columns))})",
                [record[c] for c in columns],
            )
        except Exception:
            return False


# Module-level singleton used by the app.
store = Store()
