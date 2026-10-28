"""Evaluation metrics for binary document fraud detection.

Computes Accuracy, Precision, Recall, F1-score, and AUROC from raw
prediction lists — no sklearn dependency required.

Positive class (fraudulent) convention:
  label "fraudulent" → y = 1
  label "authentic"  → y = 0
"""
from __future__ import annotations

from typing import List

import numpy as np


def compute_metrics(
    labels: List[str],
    predictions: List[str],
    probabilities: List[float],
    positive_class: str = "fraudulent",
) -> dict:
    """Compute Accuracy, Precision, Recall, F1, and AUROC.

    Args:
        labels:        Ground-truth labels ("authentic" or "fraudulent").
        predictions:   Model predicted labels (same format).
        probabilities: Model's confidence score for the positive class (0–1).
        positive_class: The class treated as positive (default: "fraudulent").

    Returns:
        dict with keys: accuracy, precision, recall, f1, auroc, tp, fp, fn, tn.
    """
    if len(labels) != len(predictions) or len(labels) != len(probabilities):
        raise ValueError("labels, predictions and probabilities must have the same length.")

    y_true  = np.array([1 if l == positive_class else 0 for l in labels], dtype=np.int32)
    y_pred  = np.array([1 if p == positive_class else 0 for p in predictions], dtype=np.int32)
    y_score = np.array(probabilities, dtype=np.float32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    n         = len(y_true)
    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    auroc = _auroc(y_true, y_score)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auroc":     round(auroc,     4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Internal AUROC (trapezoidal rule, no external dependency)
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC via the trapezoidal rule."""
    if len(np.unique(y_true)) < 2:
        return float("nan")

    # Sort by descending score
    desc_idx       = np.argsort(y_score)[::-1]
    y_true_sorted  = y_true[desc_idx]

    n_pos = int(np.sum(y_true))
    n_neg = n_pos - n_pos + len(y_true) - n_pos   # len - n_pos

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])

    return float(np.trapz(tpr, fpr))
