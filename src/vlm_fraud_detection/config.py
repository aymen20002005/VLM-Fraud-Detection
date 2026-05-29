import os
from dataclasses import dataclass

DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024


@dataclass
class VertexAIConfig:
    project_id: str
    location: str = DEFAULT_LOCATION
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "VertexAIConfig":
        project_id = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("PROJECT_ID environment variable is required")

        return cls(
            project_id=project_id,
            location=os.getenv("LOCATION", DEFAULT_LOCATION),
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )
