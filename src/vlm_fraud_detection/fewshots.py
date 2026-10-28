"""Few-shot example selection strategies.

Implements all four methods from the paper:

  DiversitySelector        – class-balanced FPS with multiple restarts (proposed)
  SimilaritySelector       – nearest-neighbour retrieval (baseline)
  RandomSelector           – uniform random sampling (baseline)
  StratifiedRandomSelector – class-balanced random sampling (baseline)

A LabeledExample is the common data unit: (image_path, label).
Labels are always lowercase strings: "authentic" or "fraudulent".
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

import numpy as np

from .embeddings import CLIPEmbeddingExtractor, HistogramEmbeddingExtractor


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

class LabeledExample(NamedTuple):
    image_path: str
    label: str          # "authentic" | "fraudulent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_distances_to_ref(embeddings: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine distance (1 − similarity) from each row to *ref*.

    Assumes all vectors are already L2-normalised.
    """
    return 1.0 - (embeddings @ ref)


def _total_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Sum of all pairwise cosine distances (upper triangle, diagonal excluded).

    Used as the diversity score in Algorithm 1 of the paper.
    """
    sim = embeddings @ embeddings.T           # (k, k) similarity matrix
    dist = 1.0 - sim
    return float(np.sum(np.triu(dist, k=1)))


def _default_extractor():
    try:
        return CLIPEmbeddingExtractor()
    except ImportError:
        return HistogramEmbeddingExtractor()


# ---------------------------------------------------------------------------
# DiversitySelector  (Algorithm 1 – proposed method)
# ---------------------------------------------------------------------------

class DiversitySelector:
    """Class-balanced farthest-point sampling with multiple restarts.

    For a pool of labeled examples and a requested prompt size K:
      - K/2 authentic examples selected via FPS from the authentic subset
      - K/2 fraudulent examples selected via FPS from the fraudulent subset
      - R independent restarts; the run maximising total pairwise cosine
        distance is kept (Eq. 5 of the paper)

    Args:
        extractor: Embedding extractor (default: CLIP ViT-L/14).
        n_restarts: Number of random restarts (paper default: 10).
    """

    def __init__(self, extractor=None, n_restarts: int = 10) -> None:
        self.extractor = extractor or _default_extractor()
        self.n_restarts = n_restarts

    # ------------------------------------------------------------------

    def select(self, pool: List[LabeledExample], k: int) -> List[LabeledExample]:
        """Return *k* diverse examples (k/2 per class)."""
        if k % 2 != 0:
            raise ValueError(f"k must be even for class-balanced selection; got k={k}.")
        k_per_class = k // 2

        auth  = [e for e in pool if e.label == "authentic"]
        fraud = [e for e in pool if e.label == "fraudulent"]

        for cls_name, cls_pool in [("authentic", auth), ("fraudulent", fraud)]:
            if len(cls_pool) < k_per_class:
                raise ValueError(
                    f"Not enough {cls_name} examples in pool: "
                    f"have {len(cls_pool)}, need {k_per_class}."
                )

        auth_emb  = self.extractor.extract_batch([e.image_path for e in auth])
        fraud_emb = self.extractor.extract_batch([e.image_path for e in fraud])

        auth_idx  = self._fps_with_restarts(auth_emb,  k_per_class)
        fraud_idx = self._fps_with_restarts(fraud_emb, k_per_class)

        return [auth[i] for i in auth_idx] + [fraud[i] for i in fraud_idx]

    # ------------------------------------------------------------------

    def _fps_with_restarts(self, embeddings: np.ndarray, k: int) -> List[int]:
        best_indices: List[int] = []
        best_score = -np.inf

        for _ in range(self.n_restarts):
            indices = self._fps_single(embeddings, k)
            score   = _total_pairwise_cosine_distance(embeddings[indices])
            if score > best_score:
                best_score   = score
                best_indices = indices

        return best_indices

    def _fps_single(self, embeddings: np.ndarray, k: int) -> List[int]:
        """One greedy FPS pass from a random seed (Eq. 6 of the paper)."""
        n = len(embeddings)
        seed = random.randint(0, n - 1)
        selected = [seed]
        min_dists = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, k):
            last = selected[-1]
            d = _cosine_distances_to_ref(embeddings, embeddings[last])
            np.minimum(min_dists, d, out=min_dists)
            min_dists[selected] = -np.inf   # exclude already selected
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists[next_idx] = -np.inf

        return selected


# ---------------------------------------------------------------------------
# SimilaritySelector  (nearest-neighbour baseline)
# ---------------------------------------------------------------------------

class SimilaritySelector:
    """K-nearest-neighbour retrieval baseline.

    Selects the k/2 authentic and k/2 fraudulent examples from the pool
    most similar (cosine) to the query document.
    """

    def __init__(self, extractor=None) -> None:
        self.extractor = extractor or _default_extractor()

    def select(
        self,
        pool: List[LabeledExample],
        k: int,
        query_path: str,
    ) -> List[LabeledExample]:
        if k % 2 != 0:
            raise ValueError(f"k must be even; got k={k}.")
        k_per_class = k // 2

        query_emb = self.extractor.extract(query_path)

        auth  = [e for e in pool if e.label == "authentic"]
        fraud = [e for e in pool if e.label == "fraudulent"]

        return (
            self._top_k_similar(auth,  query_emb, k_per_class)
            + self._top_k_similar(fraud, query_emb, k_per_class)
        )

    def _top_k_similar(
        self,
        examples: List[LabeledExample],
        query_emb: np.ndarray,
        k: int,
    ) -> List[LabeledExample]:
        embs = self.extractor.extract_batch([e.image_path for e in examples])
        sims = embs @ query_emb                   # cosine similarity (L2-normalised)
        top_k = np.argsort(sims)[::-1][:k]
        return [examples[i] for i in top_k]


# ---------------------------------------------------------------------------
# RandomSelector  (unbalanced random baseline)
# ---------------------------------------------------------------------------

class RandomSelector:
    """Uniform random sampling baseline (k total examples, no class constraint)."""

    def select(self, pool: List[LabeledExample], k: int) -> List[LabeledExample]:
        return random.sample(pool, k)


# ---------------------------------------------------------------------------
# StratifiedRandomSelector  (class-balanced random baseline)
# ---------------------------------------------------------------------------

class StratifiedRandomSelector:
    """Class-balanced random sampling baseline (k/2 per class)."""

    def select(self, pool: List[LabeledExample], k: int) -> List[LabeledExample]:
        if k % 2 != 0:
            raise ValueError(f"k must be even; got k={k}.")
        k_per_class = k // 2

        auth  = [e for e in pool if e.label == "authentic"]
        fraud = [e for e in pool if e.label == "fraudulent"]

        for cls_name, cls_pool in [("authentic", auth), ("fraudulent", fraud)]:
            if len(cls_pool) < k_per_class:
                raise ValueError(
                    f"Not enough {cls_name} examples: have {len(cls_pool)}, need {k_per_class}."
                )

        return random.sample(auth, k_per_class) + random.sample(fraud, k_per_class)


# ---------------------------------------------------------------------------
# Legacy interface (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class FewShotSelector:
    """Legacy helper that works with unlabeled (authentic-only) image pools.

    Kept for backwards compatibility with the original CLI.
    New code should use DiversitySelector with LabeledExample lists.
    """

    def __init__(self, extractor=None) -> None:
        self.extractor = extractor or HistogramEmbeddingExtractor()

    def select_diverse_examples(self, image_paths: Iterable[str], k: int = 10) -> List[str]:
        paths = [str(Path(p).expanduser().resolve()) for p in image_paths]
        if len(paths) < k:
            raise ValueError(
                f"Dataset contains only {len(paths)} images, but k={k} was requested."
            )
        embeddings = self.extractor.extract_batch(paths)
        selected   = self._farthest_point_sampling(embeddings, k)
        return [paths[i] for i in selected]

    def _farthest_point_sampling(self, embeddings: np.ndarray, k: int) -> List[int]:
        n = len(embeddings)
        selected  = [0]
        distances = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, k):
            last = selected[-1]
            diff = embeddings - embeddings[last : last + 1]
            d    = np.linalg.norm(diff, axis=1)
            np.minimum(distances, d, out=distances)
            next_idx = int(np.argmax(distances))
            selected.append(next_idx)

        return selected

    def build_fewshot_description(self, example_paths: Iterable[str]) -> str:
        example_paths = list(example_paths)
        if not example_paths:
            return ""
        lines = ["The following are reference images of legitimate documents without fraud:"]
        for idx, p in enumerate(example_paths, start=1):
            lines.append(f"Example {idx}: {Path(p).name}")
        lines.append("Use these examples as a reference for normal document appearance.")
        return "\n".join(lines)
