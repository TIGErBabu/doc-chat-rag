"""doc-chat-rag — FastAPI entrypoint."""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest import chunk_text, load_pdf
from app.embeddings import embed
from app.generator import LLMNotConfiguredError, ensure_configured, generate_answer
from app.retrieval import retrieve
from app.store import add

logger = logging.getLogger(__name__)

app = FastAPI(title="doc-chat-rag")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(pdf: UploadFile = File(...)):
    """Upload a PDF -> chunk -> embed -> store. Returns chunk count.

    Idempotent: store.add upserts on deterministic ids, so uploading
    the same file twice just refreshes it instead of erroring.
    """
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf files are accepted")

    # pypdf needs a real file on disk, so spill the upload to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf.file.read())
        tmp_path = tmp.name
    try:
        text = load_pdf(tmp_path)
        chunks = chunk_text(text)
        vectors = embed(chunks)
        source = os.path.basename(pdf.filename)
        add(chunks, vectors, source=source)
    except Exception:
        # A bad PDF or a model/store failure shouldn't take the request
        # down with a bare 500 — log the traceback, answer with JSON.
        logger.exception("ingest failed for %s", pdf.filename)
        raise HTTPException(
            status_code=500,
            detail="ingest failed — the PDF could not be processed. "
            "The service logs have the details.",
        )
    finally:
        os.unlink(tmp_path)

    return {"source": os.path.basename(pdf.filename), "chunks": len(chunks)}


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.post("/ask")
def ask(request: AskRequest):
    """Question -> retrieve + rerank -> LLM answer with citations.

    Evidence chunks stay in the response so the answer is checkable.
    Without OPENAI_API_KEY this returns 503 with setup instructions
    instead of crashing. The key is checked before retrieval: without
    it there is no answer to generate, so fail fast — and a retrieval
    failure can never crash ahead of the setup message.
    """
    try:
        ensure_configured()
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        evidence = retrieve(request.question, k=request.k)
    except Exception:
        logger.exception("retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="retrieval is unavailable right now — the embedding/rerank "
            "models failed to load or run. Try again shortly.",
        )

    try:
        result = generate_answer(request.question, evidence)
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "question": request.question,
        "evidence": evidence,
        "answer": result["answer"],
        "citations": result["citations"],
    }


# Simple UI: one static page, served by this same service. Mounted last so
# the API routes (/health, /ingest, /ask) match first. html=True makes
# GET / serve index.html. The path is anchored to this file (not the cwd)
# so it works whether you run uvicorn from the repo root or elsewhere.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
