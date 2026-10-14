"""doc-chat-rag — FastAPI entrypoint."""
from fastapi import FastAPI

app = FastAPI(title="doc-chat-rag")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /ingest — upload pdfs, chunk, embed, store
# TODO: POST /ask — retrieve, rerank, generate answer with citations
