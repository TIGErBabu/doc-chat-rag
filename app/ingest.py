"""Document ingestion: PDF -> chunks."""
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file, page by page."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    The overlap keeps context from being cut at chunk boundaries —
    a sentence split across two chunks still appears whole in
    at least one of them. Matters later for retrieval quality.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
