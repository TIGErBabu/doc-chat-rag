"""Vector store — Chroma, embedded, zero infrastructure.

Why Chroma and not pgvector: no Postgres to run, one pip install,
persists to disk. Enough for this stage of the project. If this ever
needs concurrent writers or a real production database, swap this module
for pgvector — the interface (add/query) stays the same, callers don't care.
"""
import chromadb

_client = None


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
    return _client.get_or_create_collection("docs")


def add(chunks: list[str], vectors: list[list[float]], source: str) -> None:
    """Store chunks + their vectors, tagged with the source filename.

    Uses upsert, not add: ids are deterministic ("source:i"), so
    re-ingesting the same file updates the entries instead of
    throwing a duplicate-id error. Makes /ingest idempotent.
    """
    collection = _get_collection()
    ids = [f"{source}:{i}" for i in range(len(chunks))]
    metadatas = [{"source": source}] * len(chunks)
    collection.upsert(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)


def query(vector: list[float], n: int = 5) -> list[str]:
    """Find the n chunks most similar to the query vector."""
    return [hit["text"] for hit in query_scored(vector, n)]


def query_scored(vector: list[float], n: int = 5) -> list[dict]:
    """Same as query, but keeps the score and source for each chunk.

    Chroma returns a distance (smaller = closer). Convert it to a
    similarity score with 1 / (1 + distance) so bigger means better —
    easier to read, and it doesn't depend on which distance metric
    Chroma is configured with.
    """
    collection = _get_collection()
    results = collection.query(query_embeddings=[vector], n_results=n)
    hits = []
    for text, distance, metadata in zip(
        results["documents"][0], results["distances"][0], results["metadatas"][0]
    ):
        hits.append(
            {
                "text": text,
                "score": round(1 / (1 + distance), 4),
                "source": metadata.get("source", "unknown"),
            }
        )
    return hits
