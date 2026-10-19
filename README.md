# doc-chat-rag

Chat with your documents. Upload PDFs, ask questions, get answers with citations.

## The plan

Building a retrieval-augmented generation (RAG) app from scratch — part of my AI engineer portfolio. The goal is not just a working chatbot; it is understanding every piece: chunking, embeddings, retrieval, reranking, and how to evaluate whether the answers are actually any good.

## Architecture

```
PDFs -> chunk -> embed -> chroma
                              |
user question -> retrieve -> rerank -> LLM (gpt-4o-mini) -> answer + citations
```

API in FastAPI, served with Docker.

## Status

Day 5 — there's a UI now: one static page (`static/index.html`) served by
FastAPI itself. Upload box hits /ingest and shows the chunk count, the
question box hits /ask, and answers render with their citations listed
underneath. If `OPENAI_API_KEY` isn't set, `/ask` 503s and the page says
so plainly instead of spinning forever.

## Roadmap

- [x] Document ingestion (PDF -> chunks)
- [x] Embeddings + vector store
- [x] Retrieval + reranking
- [x] FastAPI endpoints (/ingest, /ask)
- [x] Simple UI
- [x] Dockerfile
- [x] Wire an LLM into /ask (answer + citations)
- [ ] Deploy
- [x] Eval set v1 — eyeball checks (factual-correctness scoring is a later problem)

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`POST /ingest` takes a PDF file upload, `POST /ask` takes `{"question": "...", "k": 5}` and returns the generated answer, citations, and the evidence chunks it used.

`/ask` needs an OpenAI key (generation only — retrieval works without it):

```bash
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

Without the key, `/ask` returns a 503 telling you exactly that, instead
of a traceback.

Quick check without HTTP:

```bash
python scripts/ask_check.py mydoc.pdf "what is this about?"
```

Run the v1 eval set (needs the key too):

```bash
python scripts/eval_check.py mydoc.pdf
```

## Deploying

One click on Render: dashboard -> New -> Blueprint -> point at this repo
(`render.yaml` does the rest). Then set `OPENAI_API_KEY` in the dashboard —
`/ask` answers need it; uploading, retrieval, and the UI all work without it.

Two free-tier caveats worth knowing: the disk is ephemeral, so anything
uploaded through `/ingest` is forgotten when the service restarts; and the
embedding/rerank models download on first use, so the first request after a
cold start is slow. The Dockerfile listens on the `PORT` env var Render
hands it (8000 when running locally).

Live URL: _not deployed yet_

## Key decisions so far

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), local and free. OpenAI embeddings are the production upgrade path.
- **Vector store:** Chroma (embedded, zero infra). I had pgvector in the original diagram; Chroma won because there is no Postgres to run at this stage. If this ever needs a real database, the store module's interface (add/query) stays the same.
- **Reranking:** two-stage — bi-encoder pulls 3k candidates from Chroma, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-sorts them and I keep the top k. I picked the cross-encoder over a keyword-overlap heuristic because it actually reads the question next to each chunk, and it costs no new dependency (same sentence-transformers library, tiny local model). Dead simple to explain on a whiteboard.
- **/ask returns evidence, not answers.** I would rather ship retrieval I can defend than a fake answer stitched together with string templates. The generator is the next real step.
- **Re-ingesting is idempotent:** the store upserts on deterministic ids, so uploading the same PDF twice refreshes it instead of crashing on duplicate ids. Bit me once, fixed it properly.
- **UI: one static page, no framework.** I considered Streamlit (fast to build, but it would be a second service to deploy and babysit) and React (overkill for an upload box and a question box). Serving a single hand-written HTML file through FastAPI's StaticFiles keeps the whole thing one deployable unit — same container, same port, nothing new to learn. The JS is ~80 lines of fetch calls. If this ever needs auth, chat history, or streaming answers, that's when a real frontend earns its keep.
- **File uploads need python-multipart** — FastAPI won't accept `UploadFile` without it. Learned that from the error message, not the docs.
- **Generator: OpenAI, gpt-4o-mini.** I picked OpenAI over Anthropic for one boring reason: the API is the simplest to call and the mini model is cheap enough to experiment freely. Temperature 0.2 — low enough to stick to the evidence, not so low the answers read like a robot. If I outgrow it, the generator module's interface (`generate_answer(question, evidence)`) stays the same and only the client changes.
- **Citations are parsed, not trusted.** The prompt numbers evidence chunks [1], [2], ... and demands [n] markers in the answer; then a regex pulls the markers back out and maps them to chunk sources. Markers pointing past the evidence list get dropped — a model citing [99] out of 5 chunks is hallucinating, and I don't want to launder that into a real citation.
- **The eval set proves behavior, not truth.** Six doc-agnostic questions with "what good looks like" notes — I read the outputs and judge. It checks the pipeline *behaves* (cites sources, says "I don't know" on the trap question); scoring factual correctness per document is a harder, later problem.
