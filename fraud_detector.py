import json
from vertexai import init
from vertexai.generative_models import GenerativeModel

from config import (
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from prompts import FRAUD_DETECTION_PROMPT


def detect_fraud(document_text: str) -> dict:
    # Initialize Vertex AI (uses gcloud ADC)
    init(project=PROJECT_ID, location=LOCATION)

    model = GenerativeModel(MODEL_NAME)

    prompt = FRAUD_DETECTION_PROMPT.format(document_text=document_text)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "error": "Model did not return valid JSON",
            "raw_response": response.text,
        }
