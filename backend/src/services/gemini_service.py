"""
Google Gemini API service wrapper for audio, vision, and text processing.

This module exposes a thin client used by higher-level agents and services.
For translation-specific configuration (prompt templates, model variants),
see ``gemini_translation.py`` in the same package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import google.genai as genai
import os


class GeminiClient:
    """Lightweight wrapper around the Google Gemini Python client."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        transport: str = "grpc",
    ) -> None:
        """
        Initialize a Gemini client for a given model.

        Parameters
        ----------
        model_name:
            The Gemini model name, for example ``"gemini-1.5-flash"``.
        api_key:
            Optional explicit API key. If omitted, the client will fall back to
            standard Google authentication (e.g. ADC on GCP).
        transport:
            Client transport type. ``"grpc"`` is recommended for most use cases.
        """

        self._model_name = model_name
        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(
            api_key=effective_key,
            http_options={"api_version": "v1"},
        )
        self._transport = transport

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 256,
        extra_generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text from a prompt using the configured Gemini model.

        Parameters
        ----------
        prompt:
            The text prompt to send to the model.
        temperature:
            Sampling temperature; 0.0 for deterministic behaviour.
        top_p:
            Nucleus sampling parameter.
        max_output_tokens:
            Maximum number of tokens in the generated response.
        extra_generation_config:
            Optional additional generation parameters passed through to the API.

        Returns
        -------
        str
            The model's textual response content.
        """

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }

        if extra_generation_config:
            generation_config.update(extra_generation_config)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=generation_config,
        )

        # The google-genai client returns a response with candidates; we take
        # the first candidate's text content for this helper.
        if not response.candidates:
            return ""

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ""

        # Concatenate all text parts into a single string.
        text_parts = [
            part.text for part in candidate.content.parts if getattr(part, "text", "")
        ]
        return "".join(text_parts).strip()
