"""Document embedding extractors.

Default: CLIP ViT-L/14 (768-dim, L2-normalised, cosine-distance ready).
Fallback: colour histogram when torch/clip are not available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image

EmbeddingModel = Literal["clip", "layoutlm", "dit", "histogram"]


# ---------------------------------------------------------------------------
# CLIP ViT-L/14
# ---------------------------------------------------------------------------

class CLIPEmbeddingExtractor:
    """CLIP ViT-L/14 vision encoder — 768-dimensional L2-normalised embeddings.

    Paper default: images resized to 336×336, cosine distance.
    Requires: pip install openai-clip torch
    """

    _MODEL_NAME = "ViT-L/14@336px"

    def __init__(self) -> None:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "Install CLIP and PyTorch to use CLIP embeddings:\n"
                "  pip install openai-clip torch"
            ) from exc

        self._clip = clip
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self._MODEL_NAME, device=self.device)
        self.model.eval()

    def extract(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            tensor = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_batch(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack([self.extract(p) for p in image_paths])


# ---------------------------------------------------------------------------
# Colour-histogram fallback (no deep-learning dependency)
# ---------------------------------------------------------------------------

class HistogramEmbeddingExtractor:
    """Lightweight colour-histogram baseline — requires only PIL and NumPy."""

    def __init__(self, size: tuple[int, int] = (128, 128), bins_per_channel: int = 16) -> None:
        self.size = size
        self.bins_per_channel = bins_per_channel

    def extract(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            array = np.asarray(img.convert("RGB").resize(self.size), dtype=np.float32) / 255.0

        histograms = [
            np.histogram(array[:, :, c], bins=self.bins_per_channel, range=(0.0, 1.0), density=True)[0]
            for c in range(3)
        ]
        embedding = np.concatenate(histograms).astype(np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def extract_batch(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack([self.extract(p) for p in image_paths])


# Backwards-compatible alias
ImageEmbeddingExtractor = HistogramEmbeddingExtractor


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_extractor(model: EmbeddingModel = "clip"):
    """Return the embedding extractor for *model*.

    "clip"      → CLIPEmbeddingExtractor (paper default, requires torch)
    "histogram" → HistogramEmbeddingExtractor (no extra deps, for testing)
    "layoutlm" / "dit" → not yet implemented; raises NotImplementedError
    """
    if model == "clip":
        return CLIPEmbeddingExtractor()
    if model == "histogram":
        return HistogramEmbeddingExtractor()
    if model in ("layoutlm", "dit"):
        raise NotImplementedError(
            f"Extractor for '{model}' is not yet implemented. Use 'clip' or 'histogram'."
        )
    raise ValueError(f"Unknown embedding model: '{model}'. "
                     f"Choose from: clip, histogram, layoutlm, dit.")
