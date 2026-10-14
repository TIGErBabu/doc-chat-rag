"""Scratch script — proves the pipeline works end to end.

Ingest a PDF -> chunk -> embed -> store -> ask a question -> print chunks.

Not part of the app, just a sanity check. Run from the repo root:
    python scripts/sanity_check.py <path-to-pdf> "your question here"
"""
import sys

from app.ingest import load_pdf, chunk_text
from app.embeddings import embed
from app.store import add, query


def main() -> None:
    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "what is this about?"

    text = load_pdf(pdf_path)
    print(f"loaded {len(text)} chars from {pdf_path}")

    chunks = chunk_text(text)
    print(f"split into {len(chunks)} chunks")

    vectors = embed(chunks)
    print(f"embedded into {len(vectors[0])}-dim vectors")

    add(chunks, vectors, source=pdf_path)
    print("stored in chroma")

    q_vector = embed([question])[0]
    hits = query(q_vector, n=3)
    print(f"\ntop 3 chunks for: {question!r}")
    for i, hit in enumerate(hits, 1):
        print(f"\n--- chunk {i} ---\n{hit[:400]}")


if __name__ == "__main__":
    main()
