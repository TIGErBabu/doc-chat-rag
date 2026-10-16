"""doc-chat-rag — FastAPI entrypoint."""
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingest import chunk_text, load_pdf
from app.embeddings import embed
from app.generator import LLMNotConfiguredError, generate_answer
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
    """Question -> retrieve + rerank -> LLM answer with citations.

    Evidence chunks stay in the response so the answer is checkable.
    Without OPENAI_API_KEY this returns 503 with setup instructions
    instead of crashing — retrieval itself needs no key.
    """
    evidence = retrieve(request.question, k=request.k)
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


# TODO: simple UI — a page with an upload box and a question box
