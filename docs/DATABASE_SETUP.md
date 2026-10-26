# Database Setup (Phase 2 — persistence, observability)

The app runs fully **without** a database (stateless, in-memory). Configuring
one turns on persistent user workspaces (#9), pgvector storage (#1), and
persisted observability (#10). Until then, every storage call is a safe no-op.

The database is [Neon](https://neon.tech) serverless Postgres with pgvector.
Any Postgres with the `vector` extension works the same way.

## 1. Create the project
1. Create a project at [neon.tech](https://neon.tech).
2. In the **SQL Editor**, run [`db/schema.sql`](../db/schema.sql). It
   enables the `vector` extension and creates the `profiles`, `documents`,
   `chats`, `chunk_embeddings`, and `request_metrics` tables and the
   `match_chunks()` similarity function.

## 2. Configure the backend
Copy the connection string from the Neon dashboard (**Connect** → pooled
connection) and add it to your environment (`.env` locally, or your host's env
vars):

```bash
DATABASE_URL=postgresql://<user>:<password>@<endpoint>-pooler.<region>.aws.neon.tech/<db>?sslmode=require
```

The string contains the database password: keep it backend-only.

Install the driver (already in `requirements.txt`):

```bash
pip install "psycopg[binary]"
```

On boot, `src/storage.py` detects `DATABASE_URL` and flips `store.enabled = True`.

## 3. Backend endpoints (ready now)
- `GET /api/account/status` — is persistence on, and who's signed in
- `GET /api/workspace/documents` — the user's saved documents
- `GET /api/workspace/chats` — the user's saved chats
- `POST /api/workspace/chats` — save a chat (`{title, turns}`)

The workspace endpoints need a signed-in user, sent as
`Authorization: Bearer <token>`. They return graceful responses when
persistence is off.

## 4. Auth (not wired yet)
No auth provider is configured, so `Store.user_from_token()` returns `None`
and every request is anonymous: the workspace endpoints answer 401. Metrics
persistence works without auth.

To add sign-in later (for example with [Clerk](https://clerk.com)):

1. Add the provider's login flow to the frontend.
2. Send its session token as the `Authorization: Bearer` header on calls to
   `/api/workspace/*`.
3. Verify that token in `Store.user_from_token()` and return
   `{"id": <user id>, "email": ...}`. `user_id` columns are `text`, so string
   ids like Clerk's fit as-is.

## Note on serverless
Persistence makes the app stateful, so prefer a persistent host (Render /
Railway / Fly) over pure serverless for the backend. Neon itself is managed,
so only the FastAPI process needs a long-lived home.
