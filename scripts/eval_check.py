"""Scratch script — v1 eval set for the RAG pipeline.

Ingests a PDF, then runs a fixed set of golden questions through
retrieve -> generate and prints the answers next to "what good looks
like" notes. This is eyeball evaluation, not a scoring framework:
you read the outputs and judge whether the pipeline is behaving.

One case is a trap on purpose ("capital of France") — a good pipeline
says it doesn't know instead of answering from the model's training
data. That's the groundedness check.

Run from the repo root:
    python scripts/eval_check.py <path-to-pdf> [k]

Needs OPENAI_API_KEY in the env (generation step).
Not part of the app, just a scratch script.
"""
import os
import sys

from app.ingest import load_pdf, chunk_text
from app.embeddings import embed
from app.store import add
from app.retrieval import retrieve
from app.generator import LLMNotConfiguredError, generate_answer

# Golden questions. Doc-agnostic on purpose — they work on any PDF,
# and "good looks like" describes behavior, not facts. v1 checks that
# the pipeline behaves; checking factual correctness per document
# is a later, harder eval problem.
EVAL_CASES = [
    {
        "question": "What is this document about?",
        "good_looks_like": "2-4 sentence summary; cites at least one chunk; no invented details.",
    },
    {
        "question": "What are the main topics or points covered?",
        "good_looks_like": "A short grounded list; every item traceable to a cited chunk.",
    },
    {
        "question": "Summarize the document in one sentence.",
        "good_looks_like": "Actually one sentence; still cites its source chunk(s).",
    },
    {
        "question": "What dates, numbers, or names appear in the document?",
        "good_looks_like": "Extracts what's there with citations, or honestly says none were found.",
    },
    {
        "question": "What is the capital of France?",
        "good_looks_like": (
            "TRAP: must NOT answer from general knowledge. Good = "
            '"I don\'t know based on the provided documents."'
        ),
    },
    {
        "question": "What does the document say about its own limitations?",
        "good_looks_like": (
            "Either cites a real limitations passage, or hedges honestly. "
            "Watch for confident-sounding invention here."
        ),
    },
]


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/eval_check.py <path-to-pdf> [k]')
        sys.exit(1)
    pdf_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not os.path.isfile(pdf_path):
        print(f"no such file: {pdf_path}")
        sys.exit(1)

    # ingest side — mirrors POST /ingest
    source = os.path.basename(pdf_path)
    chunks = chunk_text(load_pdf(pdf_path))
    add(chunks, embed(chunks), source=source)
    print(f"ingested {len(chunks)} chunks from {source}\n")

    passed_key_check = True
    for i, case in enumerate(EVAL_CASES, 1):
        print(f"=== case {i}/{len(EVAL_CASES)}: {case['question']!r} ===")
        try:
            evidence = retrieve(case["question"], k=k)
            result = generate_answer(case["question"], evidence)
        except LLMNotConfiguredError as e:
            if passed_key_check:
                print(f"\nSTOPPED: {e}\n")
                passed_key_check = False
            break
        print(f"\nanswer: {result['answer']}")
        cited = ", ".join(
            f"[{c['chunk']}] {c['source']}" for c in result["citations"]
        )
        print(f"citations: {cited or '(none)'}")
        print(f"good looks like: {case['good_looks_like']}\n")


if __name__ == "__main__":
    main()
