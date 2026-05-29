from pathlib import Path
from typing import Iterable, List, Optional
import numpy as np
from .embeddings import ImageEmbeddingExtractor


class FewShotSelector:
    """Select a small set of diverse non-fraud examples for few-shot prompts."""

    def __init__(self, extractor: Optional[ImageEmbeddingExtractor] = None):
        self.extractor = extractor or ImageEmbeddingExtractor()

    def select_diverse_examples(self, image_paths: Iterable[str], k: int = 10) -> List[str]:
        paths = [Path(p).expanduser().resolve() for p in image_paths]
        if len(paths) < k:
            raise ValueError(f"Dataset contains only {len(paths)} images, but k={k} was requested.")

        embeddings = self.extractor.extract_batch([str(p) for p in paths])
        selected_indices = self._farthest_point_sampling(embeddings, k)
        return [str(paths[i]) for i in selected_indices]

    def _farthest_point_sampling(self, embeddings: np.ndarray, k: int) -> List[int]:
        n = len(embeddings)
        selected = [0]
        distances = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, k):
            last = selected[-1]
            diff = embeddings - embeddings[last : last + 1]
            distance_to_last = np.linalg.norm(diff, axis=1)
            distances = np.minimum(distances, distance_to_last)
            next_idx = int(np.argmax(distances))
            selected.append(next_idx)

        return selected

    def build_fewshot_description(self, example_paths: Iterable[str]) -> str:
        example_paths = list(example_paths)
        if not example_paths:
            return ""
        lines = [
            "The following are reference images of legitimate documents without fraud:",
        ]
        for idx, example_path in enumerate(example_paths, start=1):
            lines.append(f"Example {idx}: {Path(example_path).name}")
        lines.append("Use these examples as a reference for normal document appearance.")
        return "\n".join(lines)
