# doc-chat-rag

Chat with your documents. Upload PDFs, ask questions, get answers with citations.

## The plan

Building a retrieval-augmented generation (RAG) app from scratch — part of my AI engineer portfolio. The goal is not just a working chatbot; it is understanding every piece: chunking, embeddings, retrieval, reranking, and how to evaluate whether the answers are actually any good.

## Architecture

```
PDFs -> chunk -> embed -> chroma
                              |
user question -> retrieve -> rerank -> LLM (gemini-3.8-flash) -> answer + citations
```

API in FastAPI, served with Docker.

## Status

Done, and live: https://doc-chat-rag.onrender.com — the full pipeline
runs end to end on Render's free tier. Upload a PDF, ask a question, get
a grounded answer with citations. Verified live, not just locally.

Getting there took one real fight. The first deploy 502'd on every
model-loading request: torch plus the two transformer models peaked at
768 MB, and the free tier gives you 512 MB. The fix was a runtime swap,
not a bigger instance — embeddings and reranking now run on fastembed
(ONNX builds of the same models), peaking around 385 MB, and the models
are baked into the Docker image at build time, so a cold start never
downloads anything. Generation runs on Google's Gemini free tier through
its OpenAI-compatible endpoint (see the generator decision below).
Total running cost: $0.

Free-tier caveats that remain: the service sleeps when idle, so the
first hit after a quiet spell takes ~50s to wake up; and the disk is
ephemeral, so uploaded PDFs are forgotten when the service restarts.

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
- [x] Deploy — live on Render's free tier
- [x] Eval set v1 — eyeball checks (factual-correctness scoring is a later problem)

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`POST /ingest` takes a PDF file upload, `POST /ask` takes `{"question": "...", "k": 5}` and returns the generated answer, citations, and the evidence chunks it used.

`/ask` needs a Gemini API key (generation only — retrieval works
without it). Keys are free from Google AI Studio
(https://aistudio.google.com) — no credit card, no prepaid credits:

```bash
export GEMINI_API_KEY=...
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
(`render.yaml` does the rest). Then set `GEMINI_API_KEY` in the dashboard —
`/ask` answers need it; uploading, retrieval, and the UI all work without it.

Two free-tier caveats worth knowing: the disk is ephemeral, so anything
uploaded through `/ingest` is forgotten when the service restarts; and the
service sleeps when idle, so the first request after a quiet spell waits
~50s for it to wake. The models are baked into the image at build time,
so waking up never means downloading them. The Dockerfile listens on the
`PORT` env var Render hands it (8000 when running locally).

Live URL: https://doc-chat-rag.onrender.com — verified end to end:
`/health`, the UI page, `/ingest` (PDF in, chunks stored), and `/ask`
(grounded answer with citations) all work on the free tier.

## Key decisions so far

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), local and free. OpenAI embeddings are the production upgrade path.
- **Vector store:** Chroma (embedded, zero infra). I had pgvector in the original diagram; Chroma won because there is no Postgres to run at this stage. If this ever needs a real database, the store module's interface (add/query) stays the same.
- **Reranking:** two-stage — bi-encoder pulls 3k candidates from Chroma, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-sorts them and I keep the top k. I picked the cross-encoder over a keyword-overlap heuristic because it actually reads the question next to each chunk, and it costs no new dependency (same sentence-transformers library, tiny local model). Dead simple to explain on a whiteboard.
- **/ask returns evidence, not answers.** I would rather ship retrieval I can defend than a fake answer stitched together with string templates. The generator is the next real step.
- **Re-ingesting is idempotent:** the store upserts on deterministic ids, so uploading the same PDF twice refreshes it instead of crashing on duplicate ids. Bit me once, fixed it properly.
- **UI: one static page, no framework.** I considered Streamlit (fast to build, but it would be a second service to deploy and babysit) and React (overkill for an upload box and a question box). Serving a single hand-written HTML file through FastAPI's StaticFiles keeps the whole thing one deployable unit — same container, same port, nothing new to learn. The JS is ~80 lines of fetch calls. If this ever needs auth, chat history, or streaming answers, that's when a real frontend earns its keep.
- **File uploads need python-multipart** — FastAPI won't accept `UploadFile` without it. Learned that from the error message, not the docs.
- **Generator: Gemini (gemini-3.8-flash), through Google's OpenAI-compatible endpoint.** It started as OpenAI's gpt-4o-mini — the simplest API to call, and the mini model was cheap enough to experiment freely. That worked until the deployed app hit OpenAI's paywall: the API wants prepaid credits before the first call, and this project's budget is $0. Gemini's free tier doesn't (an AI Studio key, no card). And because Google exposes an OpenAI-compatible endpoint, the swap was three values — base_url, key, model name — inside the same OpenAI SDK; the chat.completions call itself never changed. Temperature 0.2 — low enough to stick to the evidence, not so low the answers read like a robot. The generator module's interface (`generate_answer(question, evidence)`) stays the same, so swapping providers again is still a client-only change.
- **Citations are parsed, not trusted.** The prompt numbers evidence chunks [1], [2], ... and demands [n] markers in the answer; then a regex pulls the markers back out and maps them to chunk sources. Markers pointing past the evidence list get dropped — a model citing [99] out of 5 chunks is hallucinating, and I don't want to launder that into a real citation.
- **The eval set proves behavior, not truth.** Six doc-agnostic questions with "what good looks like" notes — I read the outputs and judge. It checks the pipeline *behaves* (cites sources, says "I don't know" on the trap question); scoring factual correctness per document is a harder, later problem.
