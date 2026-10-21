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

The full pipeline is built and tested locally: PDF upload -> chunk ->
embed -> Chroma, two-stage retrieval with reranking, grounded answers
with citations, and a one-page UI — all served from a single container.
Deployed on Render's free tier: https://doc-chat-rag.onrender.com serves
/health and the UI page. The model-loading endpoints (`/ingest`, `/ask`)
hang and then 502 on the free tier — 512 MB isn't enough for torch plus
the transformer models, and the whole container restarts (~5 min down).
So the live URL is a UI demo for now; the pipeline itself runs locally.
Fixing this means a bigger instance or a model diet — decision pending.

## Evaluation

The v1 eval set is six doc-agnostic questions (`scripts/eval_check.py`),
run over any PDF. Five check the pipeline behaves on grounded
questions — a short summary, the main topics, a one-sentence summary,
extracted dates/names/numbers, and honest hedging when the document has
no limitations passage to quote. The sixth is a trap: "What is the
capital of France?" must come back "I don't know based on the provided
documents" instead of an answer from the model's training data — that's
the groundedness check. There is no scorer; I read the outputs and
judge. It proves the pipeline behaves, not that it's factually right —
scoring factual correctness per document is the next eval problem.

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

Deployed on Render via dashboard -> New -> Blueprint pointed at this repo
(`render.yaml` does the rest). Then set `OPENAI_API_KEY` in the dashboard —
`/ask` answers need it; uploading, retrieval, and the UI all work without it.

Two free-tier caveats worth knowing: the disk is ephemeral, so anything
uploaded through `/ingest` is forgotten when the service restarts; and the
embedding/rerank models download on first use, so the first request after a
cold start is slow. The Dockerfile listens on the `PORT` env var Render
hands it (8000 when running locally).

Live URL: https://doc-chat-rag.onrender.com — `/health` and the UI page
work; `/ingest` and `/ask` hang then 502 on the free tier (512 MB RAM
isn't enough to load the embedding + rerank models; the container
restarts, ~5 min downtime). Until that's fixed (bigger instance or a
model diet), the live URL is a UI demo only — the pipeline runs locally.

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
