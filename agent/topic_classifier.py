"""Simple semantic topic classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import TopicDefinition
from .embeddings.base import TextEmbedder


@dataclass
class TopicMatch:
    name: str
    score: float
    folder: str


class TopicClassifier:
    """Classify documents into predefined topics using cosine similarity."""

    def __init__(self, embedder: TextEmbedder, topics: Dict[str, TopicDefinition]):
        self.embedder = embedder
        self.topics = {key: value for key, value in topics.items()}
        self._topic_embeddings = self._precompute_topic_embeddings()

    def _precompute_topic_embeddings(self) -> Dict[str, List[float]]:
        entries = []
        keys = []
        for key, topic in self.topics.items():
            entries.append(topic.description)
            keys.append(key)
        if not entries:
            return {}
        embeddings = self.embedder.embed(entries)
        return {keys[idx]: embeddings[idx] for idx in range(len(keys))}

    def classify(self, document_text: str, candidate_topics: Optional[Sequence[str]] = None) -> TopicMatch:
        if not self.topics:
            raise ValueError("No topics configured. Please update config/topics.yaml")

        normalized = None
        if candidate_topics:
            normalized = set()
            for topic in candidate_topics:
                key = topic.lower().strip()
                normalized.add(key)
                self._ensure_topic_exists(key, topic)

        available = []
        topic_entries = []
        for key, topic in self.topics.items():
            if normalized and key not in normalized:
                continue
            available.append(topic)
            topic_entries.append(self._topic_embeddings[key])

        if not available:
            raise ValueError("None of the requested topics exist in the configuration.")

        doc_embedding = self.embedder.embed([document_text])[0]
        best_score = -1.0
        best_topic = available[0]
        for topic, topic_embedding in zip(available, topic_entries):
            score = _cosine_similarity(doc_embedding, topic_embedding)
            if score > best_score:
                best_topic = topic
                best_score = score

        return TopicMatch(name=best_topic.name, score=best_score, folder=best_topic.folder)

    def _ensure_topic_exists(self, normalized_key: str, display_name: str) -> None:
        if normalized_key in self.topics:
            return
        topic = TopicDefinition(name=display_name, description=display_name, folder=display_name)
        self.topics[normalized_key] = topic
        embedding = self.embedder.embed([topic.description])[0]
        self._topic_embeddings[normalized_key] = embedding


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
