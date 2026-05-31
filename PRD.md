
## `PRD.md`

```md
# Product Requirements Document

## Product Name

AI RAG Research Assistant

## Problem Statement

Reading and searching long PDFs manually is slow. Keyword search often misses context, and generic chatbots may hallucinate without showing sources.

## Goal

Build an AI assistant that allows users to upload PDFs and ask questions. The assistant should generate grounded answers using retrieved document context and display citations.

## Target Users

- Students
- Researchers
- Engineers
- Interview preparation candidates
- Recruiters evaluating AI engineering projects

## Core User Flow

```text
User uploads PDF
-> System extracts text
-> System chunks content
-> System generates embeddings
-> System stores vectors in FAISS
-> User asks a question
-> System retrieves relevant chunks
-> Gemini generates answer
-> App displays answer, citations, confidence, and source mix
