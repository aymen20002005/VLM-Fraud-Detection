from pathlib import Path
from typing import Sequence
from PIL import Image
import numpy as np


class ImageEmbeddingExtractor:
    """Extract a lightweight visual embedding for PNG images."""

    def __init__(self, size: tuple[int, int] = (128, 128), bins_per_channel: int = 16):
        self.size = size
        self.bins_per_channel = bins_per_channel

    def extract(self, image_path: str) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB").resize(self.size)
            array = np.asarray(image, dtype=np.float32) / 255.0

        histograms = [
            np.histogram(array[:, :, channel], bins=self.bins_per_channel, range=(0.0, 1.0), density=True)[0]
            for channel in range(3)
        ]
        embedding = np.concatenate(histograms, axis=0)
        embedding = embedding / np.linalg.norm(embedding.astype(np.float32) + 1e-8)
        return embedding

    def extract_batch(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack([self.extract(path) for path in image_paths])
