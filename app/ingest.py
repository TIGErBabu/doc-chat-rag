"""Document ingestion: PDF -> chunks. Built in Phase 1 (Nov 2026)."""


def load_pdf(path: str) -> str:
    # TODO: pypdf text extraction
    raise NotImplementedError


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    # TODO: decide chunking strategy — fixed size vs semantic
    raise NotImplementedError
