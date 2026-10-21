FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY static/ ./static/
# Bake the models into the image: run one embed + one rerank now, at build
# time, so both ONNX models download into model_cache/ inside the image.
# A deployed container then never downloads (or pays a first-request
# download + load) — on a 512 MB free tier that download inside a request
# is what used to kill the container.
RUN python -c "from app.embeddings import embed; embed(['warmup']); from app.retrieval import rerank; rerank('warmup', [{'text': 'warmup', 'score': 0.0, 'source': 'build'}], 1)"
# Hosts like Render hand us a PORT env var; default keeps local docker runs working.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
