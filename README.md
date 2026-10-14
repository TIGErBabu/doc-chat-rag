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

Day 1 — scaffolding. The RAG pipeline itself gets built Nov–Dec 2026.

## Roadmap

- [ ] Document ingestion (PDF -> chunks)
- [ ] Embeddings + vector store
- [ ] Retrieval + reranking
- [ ] FastAPI endpoints + simple UI
- [ ] Dockerfile + deploy
- [ ] Eval set — do the answers hold up?
