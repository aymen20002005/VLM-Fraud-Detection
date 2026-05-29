# VLM Fraud Detection

A Python library to detect fraud on PNG document images using Google Vertex AI and a few-shot selection pipeline for legitimate examples.

## Overview

This package is designed to support research-grade image fraud detection with GCP authentication. It includes:
- multimodal fraud detection for PNG document images
- a few-shot selection pipeline for legitimate non-fraud examples
- a diversity-based selection strategy using image embeddings
- support for Google Cloud Project authentication via environment variables

## Package Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure your environment:
```bash
cp .env.example .env
```

Edit `.env` with your Google Cloud settings:
```bash
PROJECT_ID=your-gcp-project-id
LOCATION=us-central1
MODEL_NAME=gemini-2.5-pro
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=1024
```

Example usage:
```python
from vlm_fraud_detection import FraudDetector, FewShotSelector

selector = FewShotSelector()
nonfraud_dataset = [
    "dataset/findit2/val/X00016469619.png",
    "dataset/findit2/val/X00016469620.png",
    # ...
]
selected_examples = selector.select_diverse_examples(nonfraud_dataset, k=10)

detector = FraudDetector()
result = detector.detect_image(
    image_path="dataset/findit2/test/X00016469619.png",
    fewshot_example_paths=selected_examples,
)
print(result)
```

## CLI Usage

Run the package from the command line:
```bash
python main.py --image path/to/document.png --dataset-dir path/to/nonfraud-dataset --fewshot-count 10
```

Or as a module:
```bash
python -m vlm_fraud_detection --image path/to/document.png --dataset-dir path/to/nonfraud-dataset
```

## Few-shot Selection

The few-shot pipeline selects representative legitimate examples using these steps:
1. compute fixed-size embeddings for each PNG image
2. measure Euclidean distances between embeddings
3. choose a set of examples that are far apart in embedding space

This helps the model see a broader range of normal documents and improves non-fraud calibration.

## Project Structure

```
.
├── pyproject.toml
├── requirements.txt
├── main.py
├── src/
│   └── vlm_fraud_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── detector.py
│       ├── embeddings.py
│       ├── fewshots.py
│       ├── prompts.py
│       └── cli.py
├── .env.example
└── dataset/
```

## Notes

- This library currently supports PNG images only.
- The GCP account must be configured with Application Default Credentials or equivalent Vertex AI access.

## References

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Documentation](https://ai.google.dev/)
