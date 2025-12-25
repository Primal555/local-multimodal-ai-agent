"""CLIP based multimodal embedding model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import MultimodalEmbedder


class ClipMultimodalEmbedder(MultimodalEmbedder):
    """Use HuggingFace CLIP checkpoints for text-image embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str]) -> List[List[float]]:
        inputs = self.processor(text=list(texts), padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        return self._normalize(outputs).cpu().tolist()

    @torch.no_grad()
    def embed_images(self, image_paths: Sequence[str]) -> List[List[float]]:
        images = [Image.open(Path(path)).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        return self._normalize(outputs).cpu().tolist()

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
