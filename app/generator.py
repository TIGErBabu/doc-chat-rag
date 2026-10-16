"""LLM generator — turning retrieved evidence into an answered question.

This is the "G" in RAG. Retrieval finds the chunks; this module asks an
LLM to answer the question using ONLY those chunks, with citations.

Two anti-hallucination rules baked into the prompt:
1. Answer only from the evidence. If it's not there, say so.
2. Every factual claim gets a [n] citation pointing at an evidence chunk.

Citations are parsed back out of the answer text with a regex and mapped
to the chunk's source, so the API response carries real, checkable
citations — not just vibes.
"""
import os
import re


class LLMNotConfiguredError(Exception):
    """Raised when there's no API key to call the LLM with.

    Carries setup instructions in the message itself, so the caller can
    hand it straight to the user instead of dumping a traceback.
    """


_client = None


def _get_client():
    # Lazy, same pattern as embeddings/store — only build the client
    # when someone actually asks a question.
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not set. /ask needs it to generate answers. "
                "Get a key at https://platform.openai.com/api-keys and set it "
                "with: export OPENAI_API_KEY=sk-... "
                "(Retrieval still works without it — only answer generation is blocked.)"
            )
        from openai import OpenAI

        _client = OpenAI(api_key=api_key)
    return _client


def _build_prompt(question: str, evidence: list[dict]) -> list[dict]:
    """Shape the evidence into a chat prompt the model can't wiggle out of.

    Evidence chunks are numbered [1], [2], ... and the system prompt
    demands citations in that format. Numbering is what makes the
    citation parsing below possible.
    """
    chunks = []
    for i, hit in enumerate(evidence, 1):
        chunks.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    evidence_block = "\n\n".join(chunks)

    return [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the evidence below. "
                "Do not use any outside knowledge. "
                "Cite every factual claim with [n] markers pointing at the "
                "evidence chunk it came from, e.g. [1], [2]. "
                "If the evidence does not contain the answer, say "
                '"I don\'t know based on the provided documents." '
                "Keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": f"Evidence:\n{evidence_block}\n\nQuestion: {question}",
        },
    ]


def _parse_citations(answer: str, evidence: list[dict]) -> list[dict]:
    """Pull [n] markers out of the answer and resolve them to chunks.

    Returns e.g. [{"chunk": 2, "source": "paper.pdf", "excerpt": "..."}].
    Markers pointing past the evidence list are ignored — a model that
    cites [99] when there are 5 chunks is making things up, and we don't
    want to launder that into a real citation.
    """
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = []
    for n in numbers:
        if 1 <= n <= len(evidence):
            chunk = evidence[n - 1]
            citations.append(
                {
                    "chunk": n,
                    "source": chunk["source"],
                    "excerpt": chunk["text"][:200],
                }
            )
    return citations


def generate_answer(question: str, evidence: list[dict]) -> dict:
    """Full generation step: question + evidence -> answer + citations.

    Returns {"answer": str, "citations": [...]}. Raises
    LLMNotConfiguredError if OPENAI_API_KEY is missing.
    """
    if not evidence:
        # Nothing retrieved — don't waste an API call asking the model
        # to answer from thin air. This is also the honest answer.
        return {
            "answer": "I don't know based on the provided documents — "
            "nothing relevant was retrieved.",
            "citations": [],
        }

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=_build_prompt(question, evidence),
        temperature=0.2,  # low = stick to the evidence, don't get creative
        max_tokens=500,
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "citations": _parse_citations(answer, evidence)}
