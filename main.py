import sys
from pathlib import Path

from vlm_fraud_detection import FraudDetector, VertexAIConfig

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: image file not found at {image_path}")
            sys.exit(1)

        config = VertexAIConfig.from_env()
        detector = FraudDetector(config=config)
        result = detector.detect_image(image_path)

        print("\n=== FRAUD DETECTION RESULT ===")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print("Usage: python main.py <image_path.png>")
