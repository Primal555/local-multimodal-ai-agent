"""Document management module."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import AppConfig
from .embeddings.base import TextEmbedder
from .storage.vector_store import VectorItem, VectorStore
from .topic_classifier import TopicClassifier
from .utils.files import PDF_EXTENSIONS, compute_sha256, ensure_directory, iter_files
from .utils.pdf import extract_text_from_pdf


class DocumentManager:
    """Handle indexing, classification, and search for PDF documents."""

    def __init__(self, config: AppConfig, embedder: TextEmbedder, vector_store: VectorStore):
        self.config = config
        self.embedder = embedder
        self.library_dir = ensure_directory(config.paths.document_library)
        self.vector_store = vector_store
        self.classifier = TopicClassifier(embedder, config.topics)

    def add_document(
        self,
        pdf_path: Path,
        candidate_topics: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, str]:
        pdf_path = Path(pdf_path)
        if pdf_path.suffix.lower() not in PDF_EXTENSIONS:
            raise ValueError(f"{pdf_path} is not a supported PDF file.")

        file_hash = compute_sha256(pdf_path)
        existing = self.vector_store.find_by_metadata("sha256", file_hash)
        if existing:
            metadata = dict(existing[0])
            metadata["duplicate"] = True
            return metadata

        raw_text = extract_text_from_pdf(pdf_path)
        preview = _summarize(raw_text)
        topic_match = self.classifier.classify(preview, candidate_topics)

        target_folder = topic_match.folder
        assigned_topic = topic_match.name
        if topic_match.score < self.config.classification.threshold:
            target_folder = self.config.classification.uncategorized_folder
            assigned_topic = target_folder

        destination_dir = ensure_directory(self.library_dir / target_folder)
        destination_path = destination_dir / pdf_path.name
        if pdf_path.resolve() != destination_path.resolve():
            shutil.copy2(pdf_path, destination_path)

        embedding = self.embedder.embed([preview])[0]
        metadata = {
            "title": title or destination_path.stem,
            "path": str(destination_path),
            "topic": assigned_topic,
            "score": f"{topic_match.score:.3f}",
            "sha256": file_hash,
        }
        if assigned_topic != topic_match.name:
            metadata["predicted_topic"] = topic_match.name
        item = VectorItem(embedding=embedding, metadata=metadata, document=preview)
        self.vector_store.upsert([item])
        return metadata

    def organize_folder(self, folder: Path, candidate_topics: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(folder)
        results = []
        for pdf_file in iter_files(folder, PDF_EXTENSIONS):
            metadata = self.add_document(pdf_file, candidate_topics=candidate_topics)
            results.append(metadata)
        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        embedding = self.embedder.embed([query])[0]
        matches = self.vector_store.query([embedding], top_k=top_k)
        documents = matches.get("documents", [[]])[0]
        metadatas = matches.get("metadatas", [[]])[0]
        distances = matches.get("distances", [[]])[0]
        results = []
        threshold = self.config.retrieval.threshold
        for doc_text, metadata, distance in zip(documents, metadatas, distances):
            metadata = dict(metadata)
            metadata["preview"] = doc_text
            similarity = 1 / (1 + distance) if distance is not None else 0.0
            if similarity < threshold:
                continue
            metadata["similarity"] = f"{similarity:.3f}"
            results.append(metadata)
        return results


def _summarize(text: str, limit: int = 800) -> str:
    if not text:
        return ""
    snippet = textwrap.shorten(text, width=limit, placeholder=" ...")
    return snippet
