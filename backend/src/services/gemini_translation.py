"""
Translation-specific wrapper around the generic Gemini client.

This module loads translation model configurations and prompt templates
from the ``config/models`` and ``prompts`` directories and exposes a
simple function for translating text between languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from .gemini_service import GeminiClient
from .groq_service import GroqClient
from .hf_service import HuggingFaceClient


# In both the backend and Airflow containers the repository root is mounted at
# /workspace. Using this as the anchor makes config/prompts resolution robust
# regardless of where this module is imported from (backend app vs. model
# pipeline scripts).
REPO_ROOT = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "models"
PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass
class TranslationModelConfig:
    """Configuration for a translation model variant."""

    id: str
    provider: str
    model_name: str
    prompt_template_id: str
    temperature: float
    top_p: float
    max_output_tokens: int


def _load_model_config(config_id: str) -> TranslationModelConfig:
    """Load a translation model configuration by identifier."""
    config_path = CONFIG_DIR / f"{config_id}.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, object] = yaml.safe_load(f)

    return TranslationModelConfig(
        id=str(raw["id"]),
        provider=str(raw.get("provider", "vertex-ai")),
        model_name=str(raw["model_name"]),
        prompt_template_id=str(raw["prompt_template_id"]),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        max_output_tokens=int(raw.get("max_output_tokens", 256)),
    )


def _load_prompt_template(template_id: str) -> str:
    """Load a prompt template by identifier."""
    template_path = PROMPTS_DIR / f"{template_id}.txt"
    with template_path.open("r", encoding="utf-8") as f:
        return f.read()


def _get_text_client(config: TranslationModelConfig):
    """
    Return a text-generation client based on the provider.

    Supported providers:
    - vertex-ai / gemini / huggingface: Hugging Face Inference API (current default)
    - groq: Groq-hosted models via OpenAI-compatible API (disabled by default)
    """
    provider = config.provider.lower()
    # For now, route vertex-ai / gemini / huggingface configs through HuggingFaceClient.
    if provider in {"vertex-ai", "gemini", "huggingface"}:
        return HuggingFaceClient(model_name=config.model_name)
    # Groq support is present but not used unless explicitly configured.
    if provider == "groq":
        return GroqClient(model_name=config.model_name)
    raise ValueError(f"Unsupported provider for translation: {config.provider}")


def translate_text(
    source_text: str,
    source_language: str,
    target_language: str,
    *,
    config_id: str = "translation_flash_v1",
) -> str:
    """
    Translate text from ``source_language`` to ``target_language`` using Gemini.

    Parameters
    ----------
    source_text:
        The input text to translate.
    source_language:
        Source language code or name (for prompt context).
    target_language:
        Target language code or name (for prompt context).
    config_id:
        Identifier of the translation model configuration to use.

    Returns
    -------
    str
        Translated text produced by the model.
    """

    config = _load_model_config(config_id)
    template = _load_prompt_template(config.prompt_template_id)

    prompt = template.format(
        source_text=source_text,
        source_language=source_language,
        target_language=target_language,
    )

    client = _get_text_client(config)
    return client.generate_text(
        prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
    )

