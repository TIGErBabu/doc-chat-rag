"""doc-chat-rag — FastAPI entrypoint."""
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingest import chunk_text, load_pdf
from app.embeddings import embed
from app.retrieval import retrieve
from app.store import add

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
    finally:
        os.unlink(tmp_path)

    return {"source": os.path.basename(pdf.filename), "chunks": len(chunks)}


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.post("/ask")
def ask(request: AskRequest):
    """Question -> retrieve + rerank -> evidence chunks with scores.

    TODO: wire the generator. Right now this returns the retrieved
    evidence only — no LLM answers anything yet. Next step is to feed
    these chunks (with their sources) into an LLM and return an answer
    with citations. Shipping the honest incremental version first.
    """
    evidence = retrieve(request.question, k=request.k)
    return {
        "question": request.question,
        "evidence": evidence,
        # TODO: replace evidence-only response with a generated answer + citations
        "answer": None,
    }


# TODO: simple UI — a page with an upload box and a question box
