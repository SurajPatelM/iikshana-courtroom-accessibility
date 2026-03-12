"""
Groq API client wrapper for text generation and translation-style prompts.

Uses the OpenAI-compatible chat completions endpoint exposed by Groq.
"""

from __future__ import annotations

import os
from typing import Optional

import requests


class GroqClient:
    """
    Minimal Groq text client with a Gemini-like generate_text interface.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in the environment "
                "before using GroqClient."
            )
        # Groq exposes an OpenAI-compatible API
        self._endpoint = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 256,
    ) -> str:
        """
        Generate text from a prompt using a Groq-hosted model.

        Parameters mirror the GeminiClient.generate_text signature where possible.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
        }

        response = requests.post(self._endpoint, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            # Fall back to raw JSON string if parsing fails
            return str(data)

