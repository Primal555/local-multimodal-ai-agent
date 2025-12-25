"""Application configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")
DEFAULT_TOPICS_PATH = Path("config/topics.yaml")


@dataclass
class PathsConfig:
    document_library: Path = Path("data/documents")
    image_library: Path = Path("data/images")
    vector_store_path: Path = Path("storage/chroma")


@dataclass
class ModelsConfig:
    text_embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"


@dataclass
class ClassificationConfig:
    threshold: float = 0.2
    uncategorized_folder: str = "Uncategorized"


@dataclass
class RetrievalConfig:
    threshold: float = 0.35
    image_threshold: float = 0.4


@dataclass
class TopicDefinition:
    name: str
    description: str
    folder: str


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    topics: Dict[str, TopicDefinition] = field(default_factory=dict)


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_topics(raw_topics: Dict[str, Dict[str, str]]) -> Dict[str, TopicDefinition]:
    parsed: Dict[str, TopicDefinition] = {}
    for name, payload in raw_topics.items():
        parsed[name.lower()] = TopicDefinition(
            name=name,
            description=payload.get("description", name),
            folder=payload.get("folder", name),
        )
    return parsed


def load_config(config_path: Optional[Path] = None, topics_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML files."""
    config_file = config_path or DEFAULT_CONFIG_PATH
    topics_file = topics_path or DEFAULT_TOPICS_PATH

    raw_config = _load_yaml(config_file)
    raw_topics = _load_yaml(topics_file).get("topics", {})

    config = AppConfig()

    paths = raw_config.get("paths", {})
    config.paths = PathsConfig(
        document_library=Path(paths.get("document_library", config.paths.document_library)),
        image_library=Path(paths.get("image_library", config.paths.image_library)),
        vector_store_path=Path(paths.get("vector_store_path", config.paths.vector_store_path)),
    )

    models = raw_config.get("models", {})
    config.models = ModelsConfig(
        text_embedding=models.get("text_embedding", config.models.text_embedding),
        clip_model=models.get("clip_model", config.models.clip_model),
    )

    classification = raw_config.get("classification", {})
    config.classification = ClassificationConfig(
        threshold=float(classification.get("threshold", config.classification.threshold)),
        uncategorized_folder=classification.get(
            "uncategorized_folder",
            config.classification.uncategorized_folder,
        ),
    )

    retrieval = raw_config.get("retrieval", {})
    config.retrieval = RetrievalConfig(
        threshold=float(retrieval.get("threshold", config.retrieval.threshold)),
        image_threshold=float(retrieval.get("image_threshold", config.retrieval.image_threshold)),
    )

    config.topics = _parse_topics(raw_topics)

    return config
