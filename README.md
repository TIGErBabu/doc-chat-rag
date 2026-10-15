# doc-chat-rag

Chat with your documents. Upload PDFs, ask questions, get answers with citations.

## The plan

Building a retrieval-augmented generation (RAG) app from scratch — part of my AI engineer portfolio. The goal is not just a working chatbot; it is understanding every piece: chunking, embeddings, retrieval, reranking, and how to evaluate whether the answers are actually any good.

## Architecture

```
PDFs -> chunk -> embed -> chroma
                              |
user question -> retrieve -> rerank -> [LLM TODO] -> answer + citations
```

API in FastAPI, served with Docker.

## Status

Day 3 — retrieval + reranking are in, and the API has `/ingest` and `/ask`. One honest caveat: `/ask` returns the retrieved evidence chunks, not a generated answer. No LLM is wired up yet, so I am shipping the retrieval half first and marking the generator as the next step instead of faking it.

## Roadmap

- [x] Document ingestion (PDF -> chunks)
- [x] Embeddings + vector store
- [x] Retrieval + reranking
- [x] FastAPI endpoints (/ingest, /ask)
- [ ] Simple UI
- [x] Dockerfile
- [ ] Wire an LLM into /ask (answer + citations)
- [ ] Deploy
- [ ] Eval set — do the answers hold up?

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`POST /ingest` takes a PDF file upload, `POST /ask` takes `{"question": "...", "k": 5}` and returns the top evidence chunks with scores.

Quick check without HTTP:

```bash
python scripts/ask_check.py mydoc.pdf "what is this about?"
```

## Key decisions so far

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), local and free. OpenAI embeddings are the production upgrade path.
- **Vector store:** Chroma (embedded, zero infra). I had pgvector in the original diagram; Chroma won because there is no Postgres to run at this stage. If this ever needs a real database, the store module's interface (add/query) stays the same.
- **Reranking:** two-stage — bi-encoder pulls 3k candidates from Chroma, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-sorts them and I keep the top k. I picked the cross-encoder over a keyword-overlap heuristic because it actually reads the question next to each chunk, and it costs no new dependency (same sentence-transformers library, tiny local model). Dead simple to explain on a whiteboard.
- **/ask returns evidence, not answers.** I would rather ship retrieval I can defend than a fake answer stitched together with string templates. The generator is the next real step.
- **Re-ingesting is idempotent:** the store upserts on deterministic ids, so uploading the same PDF twice refreshes it instead of crashing on duplicate ids. Bit me once, fixed it properly.
- **File uploads need python-multipart** — FastAPI won't accept `UploadFile` without it. Learned that from the error message, not the docs.
