"""VLM fraud detection backends.

Three backends supported:
  GeminiDetector  – Vertex AI (Gemini 2.5 Pro)     → requires: google-cloud-aiplatform, vertexai
  GPT4VDetector   – OpenAI (gpt-4o / gpt-4-vision) → requires: openai
  ClaudeDetector  – Anthropic (claude-3-5-sonnet)  → requires: anthropic

All three expose the same interface:
  detector.detect(query_path, examples, chain_of_thought) → dict

``FraudDetector`` is kept as a backwards-compatible alias for GeminiDetector.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

from .config import AnthropicConfig, OpenAIConfig, VertexAIConfig
from .fewshots import LabeledExample
from .prompts import build_text_prompt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(image_path: str) -> bytes:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()


def _to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".tif":  "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "image/png")


def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from a VLM text response."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Raw JSON object anywhere in text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "Model did not return valid JSON", "raw_response": text}


# ---------------------------------------------------------------------------
# Gemini / Vertex AI
# ---------------------------------------------------------------------------

class GeminiDetector:
    """Fraud detector backed by Vertex AI Gemini."""

    def __init__(self, config: Optional[VertexAIConfig] = None) -> None:
        self.config = config or VertexAIConfig.from_env()
        self._initialized = False

    def _bootstrap(self) -> None:
        if not self._initialized:
            from vertexai import init
            init(project=self.config.project_id, location=self.config.location)
            self._initialized = True

    def detect(
        self,
        query_path: str,
        examples: Optional[List[LabeledExample]] = None,
        chain_of_thought: bool = False,
    ) -> dict:
        self._bootstrap()
        from vertexai.generative_models import GenerativeModel, Part

        model  = GenerativeModel(self.config.model_name)
        parts: list = []

        if examples:
            for ex in examples:
                img_bytes = _load_image_bytes(ex.image_path)
                parts.append(Part.from_data(data=img_bytes, mime_type=_mime_type(ex.image_path)))

        query_bytes = _load_image_bytes(query_path)
        parts.append(Part.from_data(data=query_bytes, mime_type=_mime_type(query_path)))
        parts.append(build_text_prompt(examples, chain_of_thought=chain_of_thought))

        response = model.generate_content(
            parts,
            generation_config={
                "temperature":        self.config.temperature,
                "max_output_tokens":  self.config.max_output_tokens,
            },
        )
        return _parse_json_response(response.text)

    # ------------------------------------------------------------------
    # Legacy interface (backwards compat)
    # ------------------------------------------------------------------

    def _load_image_part(self, image_path: str):
        from vertexai.generative_models import Part
        return Part.from_data(data=_load_image_bytes(image_path), mime_type=_mime_type(image_path))

    def detect_image(
        self,
        image_path: str,
        fewshot_example_paths: Optional[Iterable[str]] = None,
        additional_prompt: Optional[str] = None,
    ) -> dict:
        examples = None
        if fewshot_example_paths:
            examples = [LabeledExample(p, "authentic") for p in fewshot_example_paths]
        return self.detect(image_path, examples=examples)

    def detect_image_with_fewshots(
        self, image_path: str, fewshot_example_paths: Iterable[str]
    ) -> dict:
        return self.detect_image(image_path=image_path, fewshot_example_paths=fewshot_example_paths)


# Backwards-compatible alias
FraudDetector = GeminiDetector


# ---------------------------------------------------------------------------
# OpenAI GPT-4V
# ---------------------------------------------------------------------------

class GPT4VDetector:
    """Fraud detector backed by OpenAI GPT-4o (gpt-4-vision)."""

    def __init__(self, config: Optional[OpenAIConfig] = None) -> None:
        self.config = config or OpenAIConfig.from_env()

    def detect(
        self,
        query_path: str,
        examples: Optional[List[LabeledExample]] = None,
        chain_of_thought: bool = False,
    ) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        client  = OpenAI(api_key=self.config.api_key)
        content = []

        if examples:
            for idx, ex in enumerate(examples, start=1):
                b64  = _to_b64(_load_image_bytes(ex.image_path))
                mime = _mime_type(ex.image_path)
                label_str = "Authentic" if ex.label == "authentic" else "Fraudulent"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                })
                content.append({"type": "text", "text": f"Example {idx}: {label_str}"})

        b64_q  = _to_b64(_load_image_bytes(query_path))
        mime_q = _mime_type(query_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_q};base64,{b64_q}", "detail": "high"},
        })
        content.append({
            "type": "text",
            "text": build_text_prompt(examples, chain_of_thought=chain_of_thought),
        })

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return _parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class ClaudeDetector:
    """Fraud detector backed by Anthropic Claude 3.5 Sonnet."""

    def __init__(self, config: Optional[AnthropicConfig] = None) -> None:
        self.config = config or AnthropicConfig.from_env()

    def detect(
        self,
        query_path: str,
        examples: Optional[List[LabeledExample]] = None,
        chain_of_thought: bool = False,
    ) -> dict:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        client  = anthropic.Anthropic(api_key=self.config.api_key)
        content = []

        if examples:
            for idx, ex in enumerate(examples, start=1):
                b64  = _to_b64(_load_image_bytes(ex.image_path))
                mime = _mime_type(ex.image_path)
                label_str = "Authentic" if ex.label == "authentic" else "Fraudulent"
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                })
                content.append({"type": "text", "text": f"Example {idx}: {label_str}"})

        b64_q  = _to_b64(_load_image_bytes(query_path))
        mime_q = _mime_type(query_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_q, "data": b64_q},
        })
        content.append({
            "type": "text",
            "text": build_text_prompt(examples, chain_of_thought=chain_of_thought),
        })

        response = client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_json_response(response.content[0].text)
