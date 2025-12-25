"""Common embedding interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence


class TextEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for each text."""


class MultimodalEmbedder(ABC):
    """Interface for embedders that support both text and images."""

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for text queries."""

    @abstractmethod
    def embed_images(self, image_paths: Sequence[str]) -> List[List[float]]:
        """Return embeddings for image files."""
