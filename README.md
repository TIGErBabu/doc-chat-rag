# doc-chat-rag

Chat with your documents. Upload PDFs, ask questions, get answers with citations.

## The plan

Building a retrieval-augmented generation (RAG) app from scratch — part of my AI engineer portfolio. The goal is not just a working chatbot; it is understanding every piece: chunking, embeddings, retrieval, reranking, and how to evaluate whether the answers are actually any good.

## Architecture (planned)

```
PDFs -> chunk -> embed -> pgvector
                              |
user question -> retrieve -> rerank -> LLM -> answer + citations
```

API in FastAPI, served with Docker.

## Status

Day 2 — ingestion, embeddings, and the vector store are in. The retrieve-and-answer flow (plus the API endpoints) comes next.

## Roadmap

- [x] Document ingestion (PDF -> chunks)
- [x] Embeddings + vector store
- [ ] Retrieval + reranking
- [ ] FastAPI endpoints + simple UI
- [x] Dockerfile
- [ ] Deploy
- [ ] Eval set — do the answers hold up?

## Key decisions so far

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), local and free. OpenAI embeddings are the production upgrade path.
- **Vector store:** Chroma (embedded, zero infra). pgvector if this ever needs a real database.
