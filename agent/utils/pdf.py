"""PDF utilities for text extraction."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path, max_pages: int | None = None) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    texts = []
    page_count = len(reader.pages)
    for idx, page in enumerate(reader.pages):
        if max_pages is not None and idx >= max_pages:
            break
        texts.append(page.extract_text() or "")
    return "\n".join(texts)
