"""Dataset loaders for fraud detection benchmarks.

Supported dataset layouts:
  1. Directory tree:  <root>/{authentic,fraudulent}/*.{png,jpg,jpeg,...}
  2. Manifest file:   each line is "<relative_image_path> <label>"
  3. CSV file:        columns "image_path" and "label"

``load_dataset`` auto-detects the format.
``split_dataset`` performs a stratified train/test split.
``subsample_pool`` returns a class-balanced subsample of size M.

The three benchmark datasets used in the paper:
  DocTamper  – document tampering (text replacement / insertion / deletion)
  MIDV-500   – identity documents (passports, IDs, driving licences)
  SROIE      – scanned receipts

Each dataset should be organised under a root directory following one of the
supported layouts above, with labels "authentic"/"fraudulent"
(or synonyms handled by LABEL_MAP below).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

from .fewshots import LabeledExample

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

LABEL_MAP: dict[str, str] = {
    # → authentic
    "authentic":  "authentic",
    "legitimate": "authentic",
    "real":       "authentic",
    "genuine":    "authentic",
    "0":          "authentic",
    "normal":     "authentic",
    # → fraudulent
    "fraudulent": "fraudulent",
    "fraud":      "fraudulent",
    "tampered":   "fraudulent",
    "fake":       "fraudulent",
    "forged":     "fraudulent",
    "1":          "fraudulent",
    "manipulated":"fraudulent",
    "spliced":    "fraudulent",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(root: str, fmt: str = "auto") -> List[LabeledExample]:
    """Load labeled document images from *root*.

    Args:
        root: Path to a directory or manifest/CSV file.
        fmt:  "auto" | "directory" | "manifest" | "csv"

    Returns:
        List of LabeledExample(image_path, label).
    """
    root_path = Path(root)
    if fmt == "auto":
        fmt = _detect_format(root_path)
    if fmt == "directory":
        return _load_directory(root_path)
    if fmt == "manifest":
        return _load_manifest(root_path)
    if fmt == "csv":
        return _load_csv(root_path)
    raise ValueError(f"Unknown dataset format: '{fmt}'. Choose from: auto, directory, manifest, csv.")


def split_dataset(
    examples: List[LabeledExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[LabeledExample], List[LabeledExample]]:
    """Stratified train / test split.

    Splits authentic and fraudulent subsets independently so that both splits
    are class-balanced.
    """
    rng = random.Random(seed)

    auth  = [e for e in examples if e.label == "authentic"]
    fraud = [e for e in examples if e.label == "fraudulent"]
    rng.shuffle(auth)
    rng.shuffle(fraud)

    n_auth_train  = int(len(auth)  * train_ratio)
    n_fraud_train = int(len(fraud) * train_ratio)

    train = auth[:n_auth_train]  + fraud[:n_fraud_train]
    test  = auth[n_auth_train:]  + fraud[n_fraud_train:]

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def subsample_pool(
    examples: List[LabeledExample],
    m: int,
    seed: int = 42,
) -> List[LabeledExample]:
    """Return a class-balanced subsample of at most *m* examples.

    If there are fewer than m/2 examples in one class the actual pool will be
    smaller than m (all available examples are included).
    """
    rng = random.Random(seed)

    auth  = [e for e in examples if e.label == "authentic"]
    fraud = [e for e in examples if e.label == "fraudulent"]

    m_per_class = m // 2
    auth_sample  = rng.sample(auth,  min(m_per_class, len(auth)))
    fraud_sample = rng.sample(fraud, min(m_per_class, len(fraud)))

    pool = auth_sample + fraud_sample
    rng.shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def _detect_format(root: Path) -> str:
    if root.is_file():
        return "csv" if root.suffix.lower() == ".csv" else "manifest"

    # Check for class-named subdirectories
    for label in LABEL_MAP:
        if (root / label).is_dir():
            return "directory"

    # Check for a manifest file
    for name in ("labels.txt", "manifest.txt", "annotations.txt"):
        if (root / name).is_file():
            return "manifest"

    # Default: walk all image files and try to infer labels from paths
    return "directory"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_directory(root: Path) -> List[LabeledExample]:
    """Walk *root* and infer labels from immediate subdirectory names."""
    examples: List[LabeledExample] = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        label = LABEL_MAP.get(subdir.name.lower())
        if label is None:
            continue
        for img_path in sorted(subdir.rglob("*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                examples.append(LabeledExample(str(img_path), label))
    return examples


def _load_manifest(path: Path) -> List[LabeledExample]:
    """Load from a text file where each line is '<image_path> <label>'."""
    if path.is_dir():
        # Try common manifest names
        for name in ("labels.txt", "manifest.txt", "annotations.txt"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No manifest file found in {path}")

    root = path.parent
    examples: List[LabeledExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            img_rel, raw_label = parts
            label = LABEL_MAP.get(raw_label.lower())
            if label is None:
                continue
            img_path = Path(img_rel)
            if not img_path.is_absolute():
                img_path = root / img_path
            examples.append(LabeledExample(str(img_path), label))
    return examples


def _load_csv(csv_path: Path) -> List[LabeledExample]:
    """Load from a CSV with columns 'image_path' (or 'image'/'path') and 'label' (or 'class')."""
    root = csv_path.parent
    examples: List[LabeledExample] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_col = row.get("image_path") or row.get("image") or row.get("path")
            lbl_col = row.get("label") or row.get("class")
            if img_col is None or lbl_col is None:
                continue
            label = LABEL_MAP.get(lbl_col.strip().lower())
            if label is None:
                continue
            img_path = Path(img_col.strip())
            if not img_path.is_absolute():
                img_path = root / img_path
            examples.append(LabeledExample(str(img_path), label))
    return examples
