"""Configuration dataclasses for all supported VLM backends.

Three backends:
  VertexAIConfig  – Gemini via Vertex AI (Google Cloud)
  OpenAIConfig    – GPT-4V / GPT-4o (OpenAI)
  AnthropicConfig – Claude 3.5 Sonnet (Anthropic)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE       = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI / Gemini
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LOCATION     = "us-central1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

# Legacy aliases kept for backwards compatibility
DEFAULT_MODEL_NAME = DEFAULT_GEMINI_MODEL


@dataclass
class VertexAIConfig:
    project_id: str
    location: str         = DEFAULT_LOCATION
    model_name: str       = DEFAULT_GEMINI_MODEL
    temperature: float    = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "VertexAIConfig":
        project_id = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("PROJECT_ID environment variable is required")
        return cls(
            project_id=project_id,
            location=os.getenv("LOCATION", DEFAULT_LOCATION),
            model_name=os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL),
            temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI GPT-4V
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GPT4V_MODEL = "gpt-4o"


@dataclass
class OpenAIConfig:
    api_key: str
    model_name: str        = DEFAULT_GPT4V_MODEL
    temperature: float     = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return cls(
            api_key=api_key,
            model_name=os.getenv("OPENAI_MODEL_NAME", DEFAULT_GPT4V_MODEL),
            temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Claude
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"


@dataclass
class AnthropicConfig:
    api_key: str
    model_name: str        = DEFAULT_CLAUDE_MODEL
    temperature: float     = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return cls(
            api_key=api_key,
            model_name=os.getenv("ANTHROPIC_MODEL_NAME", DEFAULT_CLAUDE_MODEL),
            temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )
