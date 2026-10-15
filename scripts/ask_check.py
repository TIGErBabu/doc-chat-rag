"""Scratch script — exercises ingest -> ask end to end.

Same pipeline the API endpoints run, minus HTTP. Useful for checking
retrieval + rerank quality quickly. Run from the repo root:
    python scripts/ask_check.py <path-to-pdf> "your question here" [k]

Not part of the app, just a scratch script.
"""
import os
import sys

from app.ingest import load_pdf, chunk_text
from app.embeddings import embed
from app.store import add
from app.retrieval import retrieve


def main() -> None:
    if len(sys.argv) < 3:
        print('usage: python scripts/ask_check.py <path-to-pdf> "your question here" [k]')
        sys.exit(1)
    pdf_path = sys.argv[1]
    question = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if not os.path.isfile(pdf_path):
        print(f"no such file: {pdf_path}")
        sys.exit(1)

    # ingest side — mirrors POST /ingest
    source = os.path.basename(pdf_path)
    chunks = chunk_text(load_pdf(pdf_path))
    add(chunks, embed(chunks), source=source)
    print(f"ingested {len(chunks)} chunks from {source}")

    # ask side — mirrors POST /ask
    evidence = retrieve(question, k=k)
    print(f"\ntop {len(evidence)} evidence chunks for: {question!r}")
    for i, hit in enumerate(evidence, 1):
        print(
            f"\n--- chunk {i} (rerank {hit['rerank_score']}, "
            f"retrieval {hit['retrieval_score']}, source {hit['source']}) ---"
        )
        print(hit["text"][:400])


if __name__ == "__main__":
    main()
