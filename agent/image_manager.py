"""Image indexing and search."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

from .config import AppConfig
from .embeddings.base import MultimodalEmbedder
from .storage.vector_store import VectorItem, VectorStore
from .utils.files import IMAGE_EXTENSIONS, compute_sha256, ensure_directory, iter_files


class ImageManager:
    """Index local images and support text-to-image search."""

    def __init__(self, config: AppConfig, embedder: MultimodalEmbedder, vector_store: VectorStore):
        self.config = config
        self.embedder = embedder
        self.library_dir = ensure_directory(config.paths.image_library)
        self.vector_store = vector_store

    def index_folder(self, folder: Path) -> List[Dict[str, str]]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(folder)

        indexed = []
        pending_items: List[tuple[Path, str]] = []
        batch_paths: List[Path] = []
        for image_path in iter_files(folder, IMAGE_EXTENSIONS):
            file_hash = compute_sha256(image_path)
            existing = self.vector_store.find_by_metadata("sha256", file_hash)
            if existing:
                metadata = dict(existing[0])
                metadata["duplicate"] = True
                indexed.append(metadata)
                continue

            destination = self._copy_into_library(image_path)
            batch_paths.append(destination)
            pending_items.append((destination, file_hash))
            if len(batch_paths) >= 8:
                indexed.extend(self._embed_and_store(pending_items))
                batch_paths = []
                pending_items = []

        if batch_paths:
            indexed.extend(self._embed_and_store(pending_items))
        return indexed

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        query_embedding = self.embedder.embed_text([query])[0]
        matches = self.vector_store.query([query_embedding], top_k=top_k)
        metadatas = matches.get("metadatas", [[]])[0]
        distances = matches.get("distances", [[]])[0]

        results = []
        threshold = self.config.retrieval.image_threshold
        for metadata, distance in zip(metadatas, distances):
            metadata = dict(metadata)
            similarity = 1 / (1 + distance) if distance is not None else 0.0
            if similarity < threshold:
                continue
            metadata["similarity"] = f"{similarity:.3f}"
            results.append(metadata)
        return results

    def _copy_into_library(self, source: Path) -> Path:
        destination = self.library_dir / source.name
        if destination.exists() and destination.resolve() != source.resolve():
            destination = self.library_dir / f"{source.stem}_{uuid.uuid4().hex[:6]}{source.suffix}"
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        return destination

    def _embed_and_store(self, image_items: Sequence[tuple[Path, str]]) -> List[Dict[str, str]]:
        image_paths = [path for path, _ in image_items]
        embeddings = self.embedder.embed_images([str(path) for path in image_paths])
        items = []
        indexed = []
        for (path, file_hash), embedding in zip(image_items, embeddings):
            metadata = {"path": str(path), "filename": path.name, "sha256": file_hash}
            indexed.append(metadata)
            items.append(VectorItem(embedding=embedding, metadata=metadata, document=path.name))
        self.vector_store.upsert(items)
        return indexed
