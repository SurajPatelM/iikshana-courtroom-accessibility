"""
Groq API client wrapper for text generation and translation-style prompts.

Uses the OpenAI-compatible chat completions endpoint exposed by Groq.
"""

from __future__ import annotations

import os
import time
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

        last_error = None
        for attempt in range(4):  # 4 attempts: 0, 1, 2, 3
            try:
                response = requests.post(
                    self._endpoint, headers=headers, json=body, timeout=60
                )
                if response.status_code == 429:
                    # Rate limited: wait and retry with exponential backoff
                    wait_sec = (2 ** attempt) + 2  # 3, 4, 6, 10 seconds
                    time.sleep(wait_sec)
                    last_error = response
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait_sec = (2 ** attempt) + 2
                    time.sleep(wait_sec)
                    last_error = e.response
                    continue
                raise
        else:
            if last_error is not None:
                last_error.raise_for_status()
            raise RuntimeError("Unexpected retry exhaustion")

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            # Fall back to raw JSON string if parsing fails
            return str(data)

