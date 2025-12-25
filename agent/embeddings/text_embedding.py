"""SentenceTransformer based text embedding model."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import TextEmbedder


class SentenceTransformerEmbedder(TextEmbedder):
    """Wrapper around sentence-transformers for document embeddings."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [list(vec) for vec in embeddings]
