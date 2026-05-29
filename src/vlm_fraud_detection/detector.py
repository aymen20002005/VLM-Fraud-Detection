import json
from pathlib import Path
from typing import Iterable, List, Optional

from vertexai import init
from vertexai.generative_models import GenerativeModel, Part

from .config import VertexAIConfig
from .prompts import FRAUD_DETECTION_PROMPT_TEMPLATE, FRAUD_FEWSHOT_INSTRUCTIONS


class FraudDetector:
    def __init__(self, config: Optional[VertexAIConfig] = None):
        self.config = config or VertexAIConfig.from_env()
        self._initialized = False

    def _bootstrap(self) -> None:
        if not self._initialized:
            init(project=self.config.project_id, location=self.config.location)
            self._initialized = True

    def _load_image_part(self, image_path: str) -> Part:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with image_file.open("rb") as handle:
            data = handle.read()

        return Part.from_data(data=data, mime_type="image/png")

    def detect_image(
        self,
        image_path: str,
        fewshot_example_paths: Optional[Iterable[str]] = None,
        additional_prompt: Optional[str] = None,
    ) -> dict:
        self._bootstrap()
        model = GenerativeModel(self.config.model_name)

        content_parts: List[object] = [self._load_image_part(image_path)]
        if fewshot_example_paths:
            content_parts.extend(self._load_image_part(path) for path in fewshot_example_paths)

        prompt_text = self._build_prompt(additional_prompt, bool(fewshot_example_paths))
        content_parts.append(prompt_text)

        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            },
        )

        return self._parse_response(response.text)

    def _build_prompt(self, additional_prompt: Optional[str], has_fewshots: bool) -> str:
        prompt_lines = []
        if has_fewshots:
            prompt_lines.append(FRAUD_FEWSHOT_INSTRUCTIONS)
        if additional_prompt:
            prompt_lines.append(additional_prompt)
        prompt_lines.append(FRAUD_DETECTION_PROMPT_TEMPLATE)
        return "\n\n".join(prompt_lines)

    def _parse_response(self, response_text: str) -> dict:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "error": "Model did not return valid JSON",
                "raw_response": response_text,
            }

    def detect_image_with_fewshots(self, image_path: str, fewshot_example_paths: Iterable[str]) -> dict:
        return self.detect_image(image_path=image_path, fewshot_example_paths=fewshot_example_paths)
