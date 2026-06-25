-- Supabase schema for the AI RAG Research Assistant (Phase 2: #1, #9, #10).
-- Run this in the Supabase SQL editor after creating your project.

-- pgvector for semantic search
create extension if not exists vector;

-- 1) User profiles (#9)
create table if not exists profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

-- 2) Saved documents (#9) — metadata + content hash (bytes live in Storage if desired)
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users (id) on delete cascade,
  name text not null,
  signature text not null,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  unique (user_id, signature)
);

-- 3) Saved chats (#9)
create table if not exists chats (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users (id) on delete cascade,
  title text,
  turns jsonb not null default '[]'::jsonb,
  created_at timestamptz default now()
);

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
  on chunk_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 100);

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
  user_id uuid,
  latency_ms int,
  prompt_tokens int,
  completion_tokens int,
  embedding_tokens int,
  request_cost_usd numeric,
  retrieval_mode text,
  route text,
  created_at timestamptz default now()
);

-- Row Level Security: users only see their own rows.
alter table profiles enable row level security;
alter table documents enable row level security;
alter table chats enable row level security;

create policy "own profile" on profiles for all using (auth.uid() = id) with check (auth.uid() = id);
create policy "own documents" on documents for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own chats" on chats for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
