-- Postgres schema for the AI RAG Research Assistant (Phase 2: #1, #9, #10).
-- Run this in the Neon SQL editor after creating your project.
--
-- user_id columns are plain text with no foreign key: there is no auth.users
-- table outside Supabase, and external providers (e.g. Clerk) issue string ids.
-- The backend scopes every query by user_id, so no row-level security is used.

-- pgvector for semantic search
create extension if not exists vector;

-- 1) User profiles (#9)
create table if not exists profiles (
  id text primary key,
  display_name text,
  created_at timestamptz default now()
);

-- 2) Saved documents (#9) — metadata + content hash
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  name text not null,
  signature text not null,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  unique (user_id, signature)
);

-- 3) Saved chats (#9)
create table if not exists chats (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  title text,
  turns jsonb not null default '[]'::jsonb,
  created_at timestamptz default now()
);
create index if not exists chats_user_idx on chats (user_id, created_at desc);

-- 4) Persistent embeddings for pgvector retrieval (#1)
-- 1536 dims matches openai/text-embedding-3-small.
create table if not exists chunk_embeddings (
  id bigserial primary key,
  document_signature text not null,
  chunk_id text not null,
  content text not null,
  page int,
  embedding vector(1536),
  created_at timestamptz default now()
);
create index if not exists chunk_embeddings_sig_idx on chunk_embeddings (document_signature);
create index if not exists chunk_embeddings_vec_idx
  on chunk_embeddings using hnsw (embedding vector_cosine_ops);

-- Vector similarity search used by Store.match_chunks()
create or replace function match_chunks(sig text, query_embedding vector(1536), match_count int)
returns table (chunk_id text, content text, page int, similarity float)
language sql stable as $$
  select chunk_id, content, page,
         1 - (embedding <=> query_embedding) as similarity
  from chunk_embeddings
  where document_signature = sig
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 5) Persisted observability (#10)
create table if not exists request_metrics (
  id bigserial primary key,
  user_id text,
  latency_ms int,
  prompt_tokens int,
  completion_tokens int,
  embedding_tokens int,
  request_cost_usd numeric,
  retrieval_mode text,
  route text,
  created_at timestamptz default now()
);
