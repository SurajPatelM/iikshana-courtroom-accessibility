"""
Groq API client wrapper for text generation and translation-style prompts.

Uses the OpenAI-compatible chat completions endpoint exposed by Groq.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

import requests


def _groq_backoff_seconds(response: requests.Response | None, attempt: int) -> float:
    """Sleep duration after a 429; honor Retry-After when present."""
    if response is not None:
        try:
            ra = response.headers.get("Retry-After")
        except (AttributeError, TypeError):
            ra = None
        if ra is not None and ra != "":
            try:
                return min(300.0, float(ra))
            except (ValueError, TypeError):
                pass
    # Exponential backoff capped ~90s + jitter (avoid synchronized retries)
    base = min(90.0, (2 ** min(attempt, 6)) * 1.25)
    return base + random.uniform(0.2, 2.0)


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
        system_prompt: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 256,
    ) -> str:
        """
        Generate text from a prompt using a Groq-hosted model.

        If system_prompt is provided, it is sent as a system message and prompt as user message.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
        }

        last_error: requests.Response | None = None
        response: requests.Response | None = None
        max_attempts = 12
        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    self._endpoint, headers=headers, json=body, timeout=120
                )
                if resp.status_code == 429:
                    last_error = resp
                    time.sleep(_groq_backoff_seconds(resp, attempt))
                    continue
                resp.raise_for_status()
                response = resp
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    last_error = e.response
                    time.sleep(_groq_backoff_seconds(e.response, attempt))
                    continue
                raise
        if response is None:
            if last_error is not None:
                last_error.raise_for_status()
            raise RuntimeError("Unexpected Groq retry exhaustion")

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            # Fall back to raw JSON string if parsing fails
            return str(data)

