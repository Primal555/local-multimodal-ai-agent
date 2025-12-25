"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import hashlib

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_files(root: Path, extensions: Sequence[str]) -> Iterator[Path]:
    normalized = {ext.lower() for ext in extensions}
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in normalized:
            yield file_path


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
