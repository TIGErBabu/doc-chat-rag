"""Retrieval + reranking — turning a question into ranked evidence.

Two stages, because each one is good at a different thing:

1. Bi-encoder retrieval (fast, approximate): the query is embedded once
   and Chroma returns the nearest chunks. Cheap enough to search thousands
   of chunks, but it only compares vectors — it never really "reads" the
   question next to the chunk.

2. Cross-encoder rerank (slow, precise): a small model reads the
   (question, chunk) pair together and scores how relevant the chunk is.
   Too slow to run over the whole store, perfect for re-sorting a
   shortlist. Runs on fastembed like the embedding model — the model is
   ms-marco-MiniLM-L-6-v2, via Xenova's ONNX export of it (the
   cross-encoder org's own repo has no ONNX build fastembed can load;
   same weights, lighter runtime).

So: retrieve 3x candidates with the bi-encoder, rerank them with the
cross-encoder, keep the top k. Textbook two-stage retrieval.
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.embeddings import MODEL_CACHE_DIR, embed
from app.store import query_scored

_reranker = None


def _get_reranker():
    # Lazy load, same reason as the embedding model — weights are slow.
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(
            "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=str(MODEL_CACHE_DIR)
        )
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-sort candidate chunks by cross-encoder relevance, keep top_k.

    candidates come from store.query_scored — dicts with
    text / score / source. Returns the same shape plus rerank_score,
    ordered best-first.
    """
    if not candidates:
        return []
    reranker = _get_reranker()
    scores = list(reranker.rerank(query, [c["text"] for c in candidates]))
    ranked = sorted(
        zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
    )
    results = []
    for candidate, score in ranked[:top_k]:
        results.append(
            {
                "text": candidate["text"],
                "source": candidate["source"],
                "retrieval_score": candidate["score"],
                "rerank_score": round(float(score), 4),
            }
        )
    return results


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Full retrieve + rerank pipeline: question -> top-k evidence chunks.

    Oversample (3k) from the vector store, then let the cross-encoder
    pick the real top k. k=5 keeps the evidence set small enough that
    the generator doesn't drown in context.
    """
    query_vector = embed([query])[0]
    candidates = query_scored(query_vector, n=k * 3)
    return rerank(query, candidates, top_k=k)
