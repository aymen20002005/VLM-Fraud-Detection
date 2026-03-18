import json
from pathlib import Path
from vertexai import init
from vertexai.generative_models import GenerativeModel, Part

from config import (
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from prompts import FRAUD_DETECTION_PROMPT


def detect_fraud(pdf_path: str) -> dict:
    """Detect fraud in PDF document using multimodal visual analysis."""
    # Verify PDF file exists
    if not Path(pdf_path).exists():
        return {
            "error": f"PDF file not found: {pdf_path}"
        }
    
    # Initialize Vertex AI (uses gcloud ADC)
    init(project=PROJECT_ID, location=LOCATION)

    model = GenerativeModel(MODEL_NAME)

    # Load PDF directly as Part object
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = pdf_file.read()
    
    pdf_part = Part.from_data(
        data=pdf_data,
        mime_type="application/pdf"
    )
    
    # Build multimodal content: PDF + prompt
    content_parts = [pdf_part, FRAUD_DETECTION_PROMPT]

    response = model.generate_content(
        content_parts,
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


def detect_fraud_from_text(document_text: str) -> dict:
    """Fallback function for text-only fraud detection."""
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
