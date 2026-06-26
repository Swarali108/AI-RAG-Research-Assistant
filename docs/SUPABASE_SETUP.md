# Supabase Setup (Phase 2 — persistence, auth, observability)

The app runs fully **without** Supabase (stateless, in-memory). Configuring it
turns on persistent user workspaces (#9), pgvector storage (#1), and persisted
observability (#10). Until then, every storage call is a safe no-op.

## 1. Create the project
1. Create a project at [supabase.com](https://supabase.com).
2. In **SQL Editor**, run [`db/schema.sql`](../db/schema.sql). It
   enables the `vector` extension and creates the `profiles`, `documents`,
   `chats`, `chunk_embeddings`, and `request_metrics` tables, the
   `match_chunks()` similarity function, and row-level-security policies.

## 2. Configure the backend
Add to your environment (`.env` locally, or your host's env vars):

```bash
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_KEY=<service-role key>     # backend only — never expose to the browser
# Optional: SUPABASE_JWT_SECRET=<jwt secret>  (for verifying user tokens offline)
```

Install the client (already in `requirements.txt`):

```bash
pip install supabase
```

On boot, `src/storage.py` detects these vars and flips `store.enabled = True`.

## 3. Backend endpoints (ready now)
- `GET /api/account/status` — is persistence on, and who's signed in
- `GET /api/workspace/documents` — the user's saved documents
- `GET /api/workspace/chats` — the user's saved chats
- `POST /api/workspace/chats` — save a chat (`{title, turns}`)

All require a Supabase access token in `Authorization: Bearer <token>` and
return graceful responses when persistence is off.

## 4. Frontend auth (the remaining wiring)
The backend verifies Supabase tokens; the browser side needs a small login flow:

1. Add the Supabase JS client with your project URL + **anon** key.
2. Use `supabase.auth.signInWithOtp` / `signInWithPassword` for login.
3. Send the session's `access_token` as the `Authorization: Bearer` header on
   calls to `/api/workspace/*`.

This is intentionally left as a focused follow-up so secrets and the auth UX
match your deployment choices. Everything it calls already exists server-side.

## Note on serverless
Persistence makes the app stateful, so prefer a persistent host (Render /
Railway / Fly) over pure serverless for the backend. Supabase itself is managed,
so only the FastAPI process needs a long-lived home.
