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


# In Docker containers the repository root is mounted at /workspace. When that
# path does not exist (e.g. local Python run on the host), fall back to the
# project root by walking up from backend/src/services to the repo root.
_this_file = Path(__file__).resolve()
if Path("/workspace").exists():
    REPO_ROOT = Path("/workspace")
else:
    # backend/src/services/gemini_translation.py -> repo root is parents[3]
    # .../backend/src/services -> parents[0]
    # .../backend/src         -> parents[1]
    # .../backend             -> parents[2]
    # .../                    -> parents[3]
    REPO_ROOT = _this_file.parents[3]
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
    system_prompt_id: str | None = None  # optional; if set, used as system message


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
        system_prompt_id=str(raw["system_prompt_id"]) if raw.get("system_prompt_id") else None,
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
    Translate text from ``source_language`` to ``target_language`` using the configured model.

    Supports system + user prompts when config has system_prompt_id; otherwise
    uses a single user prompt (prompt_template_id).
    """

    config = _load_model_config(config_id)
    user_template = _load_prompt_template(config.prompt_template_id)
    user_prompt = user_template.format(
        source_text=source_text,
        source_language=source_language,
        target_language=target_language,
    )
    system_prompt: str | None = None
    if config.system_prompt_id:
        system_prompt = _load_prompt_template(config.system_prompt_id)

    client = _get_text_client(config)
    return client.generate_text(
        user_prompt,
        system_prompt=system_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
    )

