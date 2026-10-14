"""Text embeddings — turning chunks into vectors.

Using sentence-transformers (all-MiniLM-L6-v2): free, runs locally,
no API key needed. Good enough for development and demos.

Production alternative: OpenAI text-embedding-3-small — better quality,
costs money per token, needs an API key. If this ever leaves the laptop,
swap the inside of embed() and nothing else changes.
"""
from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    # Lazy load — the model weights take a few seconds to load,
    # so don't pay for it at import time.
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts -> one vector per text."""
    model = _get_model()
    return model.encode(texts).tolist()
