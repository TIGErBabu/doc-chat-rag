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
    """Store chunks + their vectors, tagged with the source filename."""
    collection = _get_collection()
    ids = [f"{source}:{i}" for i in range(len(chunks))]
    metadatas = [{"source": source}] * len(chunks)
    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)


def query(vector: list[float], n: int = 5) -> list[str]:
    """Find the n chunks most similar to the query vector."""
    collection = _get_collection()
    results = collection.query(query_embeddings=[vector], n_results=n)
    return results["documents"][0]
