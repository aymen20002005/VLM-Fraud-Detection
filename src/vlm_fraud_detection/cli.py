import argparse
from pathlib import Path
from .config import DEFAULT_LOCATION, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, VertexAIConfig
from .detector import FraudDetector
from .fewshots import FewShotSelector


def _find_png_paths(directory: Path):
    return sorted(str(path) for path in directory.glob("**/*.png") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM fraud detection for PNG images")
    parser.add_argument("--image", help="PNG image file to analyze")
    parser.add_argument("--dataset-dir", help="Directory of legitimate PNG examples for few-shot selection")
    parser.add_argument("--fewshot-count", type=int, default=10, help="Number of non-fraud examples to select")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex AI location")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Vertex AI generative model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature")
    args = parser.parse_args()

    if not args.image:
        raise SystemExit("Error: --image is required")

    config = VertexAIConfig(
        project_id=args.project_id or VertexAIConfig.from_env().project_id,
        location=args.location,
        model_name=args.model_name,
        temperature=args.temperature,
    )

    selector = FewShotSelector()
    fewshot_paths = []
    if args.dataset_dir:
        image_paths = _find_png_paths(Path(args.dataset_dir))
        fewshot_paths = selector.select_diverse_examples(image_paths, k=args.fewshot_count)
        print(f"Selected {len(fewshot_paths)} few-shot examples from {args.dataset_dir}")

    detector = FraudDetector(config=config)
    result = detector.detect_image(args.image, fewshot_example_paths=fewshot_paths)

    print("\n=== FRAUD DETECTION RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")
