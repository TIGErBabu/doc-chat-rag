"""Text embeddings — turning chunks into vectors.

Using fastembed with the ONNX build of all-MiniLM-L6-v2: free, runs
locally, no API key needed. Same model as before, different runtime —
this used to be sentence-transformers (PyTorch), and torch plus the
two transformer models peaked around 770 MB RSS, which is more than
a small host (or Render's free tier) has. ONNX does the same job in
a fraction of the memory.

Production alternative: OpenAI text-embedding-3-small — better quality,
costs money per token, needs an API key. If this ever leaves the laptop,
swap the inside of embed() and nothing else changes.
"""
from pathlib import Path

from fastembed import TextEmbedding

# Model files live in one place, anchored to this file (not the cwd) —
# same trick as the Chroma dir in store.py, so the app and the scripts
# agree on it. The Dockerfile downloads the models here at build time,
# so a deployed container never downloads anything on a first request;
# locally they land here on first use instead. Gitignored either way.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "model_cache"

_model = None


def _get_model():
    # Lazy load — the model takes a moment to load,
    # so don't pay for it at import time.
    global _model
    if _model is None:
        _model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(MODEL_CACHE_DIR),
        )
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts -> one vector per text."""
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts)]
