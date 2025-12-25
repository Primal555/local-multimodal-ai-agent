"""Command line entry point for the Local Multimodal AI Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from agent.config import AppConfig, load_config
from agent.document_manager import DocumentManager
from agent.embeddings.clip_embedding import ClipMultimodalEmbedder
from agent.embeddings.text_embedding import SentenceTransformerEmbedder
from agent.image_manager import ImageManager
from agent.storage.vector_store import VectorStore


def parse_topics(topics: str | None) -> List[str] | None:
    if not topics:
        return None
    return [topic.strip() for topic in topics.split(",") if topic.strip()]


def build_document_manager(config: AppConfig) -> DocumentManager:
    embedder = SentenceTransformerEmbedder(config.models.text_embedding)
    store = VectorStore(config.paths.vector_store_path, "documents")
    return DocumentManager(config, embedder, store)


def build_image_manager(config: AppConfig) -> ImageManager:
    embedder = ClipMultimodalEmbedder(config.models.clip_model)
    store = VectorStore(config.paths.vector_store_path, "images")
    return ImageManager(config, embedder, store)


def cmd_add_paper(args: argparse.Namespace, config: AppConfig) -> None:
    manager = build_document_manager(config)
    metadata = manager.add_document(
        Path(args.path),
        candidate_topics=parse_topics(args.topics),
        title=args.title,
    )
    print("Indexed document:")
    _print_json(metadata)


def cmd_search_paper(args: argparse.Namespace, config: AppConfig) -> None:
    manager = build_document_manager(config)
    matches = manager.search(args.query, top_k=args.top_k)
    _print_json(matches)


def cmd_organize(args: argparse.Namespace, config: AppConfig) -> None:
    manager = build_document_manager(config)
    results = manager.organize_folder(
        Path(args.folder),
        candidate_topics=parse_topics(args.topics),
    )
    _print_json(results)


def cmd_index_images(args: argparse.Namespace, config: AppConfig) -> None:
    manager = build_image_manager(config)
    indexed = manager.index_folder(Path(args.folder))
    _print_json(indexed)


def cmd_search_image(args: argparse.Namespace, config: AppConfig) -> None:
    manager = build_image_manager(config)
    matches = manager.search(args.query, top_k=args.top_k)
    _print_json(matches)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent CLI")
    parser.add_argument("--config", type=Path, help="Path to custom config YAML file")
    parser.add_argument("--topics_config", type=Path, help="Path to custom topics YAML file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add_paper", help="Add and classify a single PDF paper")
    add_parser.add_argument("path", type=Path, help="Path to a PDF file")
    add_parser.add_argument("--topics", type=str, help="Comma separated topic candidates (e.g., 'CV,NLP')")
    add_parser.add_argument("--title", type=str, help="Optional human readable title")
    add_parser.set_defaults(func=cmd_add_paper)

    search_parser = subparsers.add_parser("search_paper", help="Semantic search for indexed documents")
    search_parser.add_argument("query", type=str, help="Natural language query")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of matches to return")
    search_parser.set_defaults(func=cmd_search_paper)

    organize_parser = subparsers.add_parser("organize_papers", help="Batch organize a folder of PDFs")
    organize_parser.add_argument("folder", type=Path, help="Folder containing PDFs")
    organize_parser.add_argument("--topics", type=str, help="Candidate topics to match against")
    organize_parser.set_defaults(func=cmd_organize)

    index_images = subparsers.add_parser("index_images", help="Index all images within a folder")
    index_images.add_argument("folder", type=Path, help="Folder containing images")
    index_images.set_defaults(func=cmd_index_images)

    search_images = subparsers.add_parser("search_image", help="Search images using a text prompt")
    search_images.add_argument("query", type=str, help="Natural language description")
    search_images.add_argument("--top_k", type=int, default=5)
    search_images.set_defaults(func=cmd_search_image)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(config_path=args.config, topics_path=args.topics_config)
    args.func(args, config)


if __name__ == "__main__":
    main()
