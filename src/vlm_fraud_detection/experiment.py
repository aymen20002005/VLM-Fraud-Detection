"""Full experimental pipeline for the paper:
"Few-Shot Vision-Language Models for Document Fraud Detection".

Implements the complete sweep described in Section IV of the paper:
  - Prompt sizes K ∈ {2, 4, 6, 8, 10}
  - Candidate pool sizes M ∈ {20, 50, 100, 200, 500}
  - 5 methods: zero_shot, random, stratified_random, similarity, diversity
  - 3 VLM backends: gemini, gpt4v, claude
  - Metrics: Accuracy, Precision, Recall, F1, AUROC
  - Random baseline averaged over 20 independent runs (paper convention)

Usage (from repo root, with venv activated):

  # Main experiment on a single dataset
  python -m vlm_fraud_detection.experiment \\
      --dataset-dir data/doctamper \\
      --dataset-name doctamper \\
      --vlm gemini \\
      --output-dir results/

  # Ablation: embedding model comparison
  python -m vlm_fraud_detection.experiment \\
      --dataset-dir data/doctamper \\
      --dataset-name doctamper \\
      --vlm gemini \\
      --ablation embeddings \\
      --output-dir results/ablation/

Results are written to CSV files in --output-dir.
Each row corresponds to one (K, M, method, run) combination.
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .config import AnthropicConfig, OpenAIConfig, VertexAIConfig
from .dataset import load_dataset, split_dataset, subsample_pool
from .detector import ClaudeDetector, GeminiDetector, GPT4VDetector
from .embeddings import build_extractor
from .fewshots import (
    DiversitySelector,
    LabeledExample,
    RandomSelector,
    SimilaritySelector,
    StratifiedRandomSelector,
)
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

ALL_METHODS = ["zero_shot", "random", "stratified_random", "similarity", "diversity"]
ALL_K       = [2, 4, 6, 8, 10]
ALL_M       = [20, 50, 100, 200, 500]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    dataset:          str
    vlm:              str
    embedding_model:  str
    method:           str
    k:                int
    m:                int
    run:              int
    accuracy:         float
    precision:        float
    recall:           float
    f1:               float
    auroc:            float
    n_queries:        int
    elapsed_seconds:  float


# ---------------------------------------------------------------------------
# VLM factory
# ---------------------------------------------------------------------------

def _build_detector(vlm: str):
    if vlm == "gemini":
        return GeminiDetector()
    if vlm == "gpt4v":
        return GPT4VDetector()
    if vlm == "claude":
        return ClaudeDetector()
    raise ValueError(f"Unknown VLM '{vlm}'. Choose from: gemini, gpt4v, claude.")


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    *,
    detector,
    train_pool: List[LabeledExample],
    test_set:   List[LabeledExample],
    method:     str,
    k:          int,
    m:          int,
    extractor,
    run_seed:        int   = 42,
    chain_of_thought: bool = False,
    max_retries:     int   = 3,
    retry_delay:     float = 2.0,
) -> dict:
    """Evaluate one (method, K, M, seed) configuration.

    Returns the metrics dict from ``compute_metrics``.
    """
    random.seed(run_seed)

    pool = subsample_pool(train_pool, m, seed=run_seed)

    labels:        List[str]   = []
    predictions:   List[str]   = []
    probabilities: List[float] = []

    # Pre-compute DiversitySelector support set once per (method, pool, k)
    # to avoid recomputing embeddings for every query.
    diversity_examples: Optional[List[LabeledExample]] = None
    if method == "diversity":
        diversity_examples = DiversitySelector(extractor).select(pool, k)

    for query in test_set:
        # ---- select examples ------------------------------------------------
        if method == "zero_shot":
            examples: List[LabeledExample] = []
        elif method == "random":
            examples = RandomSelector().select(pool, k)
        elif method == "stratified_random":
            examples = StratifiedRandomSelector().select(pool, k)
        elif method == "similarity":
            examples = SimilaritySelector(extractor).select(pool, k, query.image_path)
        elif method == "diversity":
            examples = diversity_examples  # reuse pre-computed set
        else:
            raise ValueError(f"Unknown method: '{method}'")

        # ---- call VLM with retries ------------------------------------------
        result: Optional[dict] = None
        for attempt in range(max_retries):
            try:
                result = detector.detect(
                    query.image_path,
                    examples=examples,
                    chain_of_thought=chain_of_thought,
                )
                break
            except Exception as exc:
                logger.warning(
                    "VLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if result is None:
            result = {
                "error":             "All retries failed",
                "fraud_label":       "Authentic",
                "fraud_probability": 0.5,
            }

        # ---- parse output ---------------------------------------------------
        raw_label = result.get("fraud_label", "Authentic")
        prob      = float(result.get("fraud_probability", 0.5))

        pred = "fraudulent" if isinstance(raw_label, str) and raw_label.lower() in (
            "fraudulent", "fraud"
        ) else "authentic"

        labels.append(query.label)
        predictions.append(pred)
        probabilities.append(prob)

    return compute_metrics(labels, predictions, probabilities)


# ---------------------------------------------------------------------------
# Full experiment sweep
# ---------------------------------------------------------------------------

def run_experiments(
    dataset_dir:      str,
    output_dir:       str,
    vlm:              str   = "gemini",
    dataset_name:     str   = "dataset",
    embedding_model:  str   = "clip",
    k_values:         Optional[List[int]] = None,
    m_values:         Optional[List[int]] = None,
    methods:          Optional[List[str]] = None,
    n_runs:           int   = 5,
    random_n_runs:    int   = 20,
    train_ratio:      float = 0.8,
    chain_of_thought: bool  = False,
    seed:             int   = 42,
) -> List[ExperimentResult]:
    """Run the full sweep described in Section IV of the paper.

    Args:
        dataset_dir:      Root directory of the labeled dataset.
        output_dir:       Directory where CSV results are saved.
        vlm:              VLM backend: "gemini", "gpt4v", or "claude".
        dataset_name:     Label used in output file names.
        embedding_model:  Embedding extractor: "clip" or "histogram".
        k_values:         Prompt sizes to sweep.  Default: [2,4,6,8,10].
        m_values:         Pool sizes to sweep.    Default: [20,50,100,200,500].
        methods:          Selection methods.       Default: all five methods.
        n_runs:           Independent runs for diversity / similarity.
        random_n_runs:    Independent runs for random baselines (paper: 20).
        train_ratio:      Fraction used as candidate pool.
        chain_of_thought: Use CoT prompting.
        seed:             Base random seed for dataset split.
    """
    k_values = k_values or ALL_K
    m_values = m_values or ALL_M
    methods  = methods  or ALL_METHODS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- data ---------------------------------------------------------------
    logger.info("Loading dataset from %s", dataset_dir)
    all_examples = load_dataset(dataset_dir)
    n_auth  = sum(1 for e in all_examples if e.label == "authentic")
    n_fraud = sum(1 for e in all_examples if e.label == "fraudulent")
    logger.info("Loaded %d examples  (authentic=%d  fraudulent=%d)",
                len(all_examples), n_auth, n_fraud)

    train_set, test_set = split_dataset(all_examples, train_ratio=train_ratio, seed=seed)
    logger.info("Split → train=%d  test=%d", len(train_set), len(test_set))

    # ---- embedding extractor ------------------------------------------------
    logger.info("Building embedding extractor: %s", embedding_model)
    extractor = build_extractor(embedding_model)

    # ---- VLM ----------------------------------------------------------------
    detector = _build_detector(vlm)

    # ---- results file -------------------------------------------------------
    results_file = output_path / f"{dataset_name}_{vlm}_{embedding_model}_results.csv"
    fieldnames   = list(ExperimentResult.__dataclass_fields__.keys())
    all_results: List[ExperimentResult] = []

    with open(results_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for k in k_values:
            for m in m_values:
                for method in methods:
                    if method == "zero_shot":
                        n_method_runs = 1
                    elif method in ("random", "stratified_random"):
                        n_method_runs = random_n_runs
                    else:
                        n_method_runs = n_runs

                    logger.info(
                        "K=%-2d  M=%-3d  method=%-18s  runs=%d",
                        k, m, method, n_method_runs,
                    )

                    for run in range(n_method_runs):
                        t0 = time.time()
                        try:
                            metrics = run_single_experiment(
                                detector=detector,
                                train_pool=train_set,
                                test_set=test_set,
                                method=method,
                                k=k,
                                m=m,
                                extractor=extractor,
                                run_seed=seed + run,
                                chain_of_thought=chain_of_thought,
                            )
                        except Exception as exc:
                            logger.error("Experiment failed: %s", exc)
                            continue

                        elapsed = time.time() - t0
                        result  = ExperimentResult(
                            dataset=dataset_name,
                            vlm=vlm,
                            embedding_model=embedding_model,
                            method=method,
                            k=k,
                            m=m,
                            run=run,
                            accuracy=metrics["accuracy"],
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                            f1=metrics["f1"],
                            auroc=metrics["auroc"],
                            n_queries=len(test_set),
                            elapsed_seconds=round(elapsed, 1),
                        )
                        all_results.append(result)
                        writer.writerow(asdict(result))
                        csvfile.flush()

                        logger.info(
                            "  run %-2d → Acc=%.3f  P=%.3f  R=%.3f  F1=%.3f  AUROC=%.3f",
                            run, result.accuracy, result.precision,
                            result.recall, result.f1, result.auroc,
                        )

    logger.info("Results saved to %s", results_file)
    return all_results


# ---------------------------------------------------------------------------
# Ablation: embedding model comparison  (Section V-C)
# ---------------------------------------------------------------------------

def run_ablation_embeddings(
    dataset_dir:  str,
    output_dir:   str,
    vlm:          str        = "gemini",
    dataset_name: str        = "dataset",
    k:            int        = 8,
    m:            int        = 100,
    n_runs:       int        = 5,
    embedding_models: Optional[List[str]] = None,
) -> List[ExperimentResult]:
    """Compare CLIP vs histogram embeddings on the diversity method."""
    embedding_models = embedding_models or ["clip", "histogram"]
    all_results: List[ExperimentResult] = []
    for emb in embedding_models:
        logger.info("Ablation — embedding: %s", emb)
        results = run_experiments(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            vlm=vlm,
            dataset_name=f"{dataset_name}_emb_{emb}",
            embedding_model=emb,
            k_values=[k],
            m_values=[m],
            methods=["diversity"],
            n_runs=n_runs,
        )
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run few-shot VLM fraud detection experiments (Section IV of the paper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir",  required=True,
                        help="Path to the labeled dataset root directory.")
    parser.add_argument("--output-dir",   default="results",
                        help="Directory where CSV results are written.")
    parser.add_argument("--dataset-name", default="dataset",
                        help="Label used in output file names (e.g. 'doctamper').")
    parser.add_argument("--vlm", default="gemini",
                        choices=["gemini", "gpt4v", "claude"],
                        help="VLM backend to use.")
    parser.add_argument("--embedding-model", default="clip",
                        choices=["clip", "histogram"],
                        help="Embedding model for few-shot selection.")
    parser.add_argument("--k-values", nargs="+", type=int, default=ALL_K,
                        help="Prompt sizes K to evaluate.")
    parser.add_argument("--m-values", nargs="+", type=int, default=ALL_M,
                        help="Candidate pool sizes M to evaluate.")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Selection methods to run.")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Independent runs for diversity/similarity methods.")
    parser.add_argument("--random-n-runs", type=int, default=20,
                        help="Independent runs for random baselines.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of examples used as candidate pool.")
    parser.add_argument("--chain-of-thought", action="store_true",
                        help="Use chain-of-thought prompting.")
    parser.add_argument("--ablation", default="none",
                        choices=["none", "embeddings"],
                        help="Run an ablation study instead of the main sweep.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.ablation == "embeddings":
        run_ablation_embeddings(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            vlm=args.vlm,
            dataset_name=args.dataset_name,
            k=8,
            m=100,
            n_runs=args.n_runs,
        )
    else:
        run_experiments(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            vlm=args.vlm,
            dataset_name=args.dataset_name,
            embedding_model=args.embedding_model,
            k_values=args.k_values,
            m_values=args.m_values,
            methods=args.methods,
            n_runs=args.n_runs,
            random_n_runs=args.random_n_runs,
            train_ratio=args.train_ratio,
            chain_of_thought=args.chain_of_thought,
        )


if __name__ == "__main__":
    main()
