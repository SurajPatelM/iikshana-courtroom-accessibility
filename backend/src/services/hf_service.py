"""
Hugging Face Inference API client wrapper for text generation / translation.

This client uses the hosted Inference API and exposes a generate_text interface
compatible with the GeminiClient / GroqClient helpers.
"""

from __future__ import annotations

import os
from typing import Optional

import requests


class HuggingFaceClient:
    """
    Minimal Hugging Face text client with a generate_text interface.
    """

    def __init__(self, model_name: str, api_token: Optional[str] = None) -> None:
        self._model_name = model_name
        self._api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "HF_API_TOKEN is not set. Please set it in the environment "
                "before using HuggingFaceClient."
            )
        self._endpoint = (
            f"https://api-inference.huggingface.co/models/{self._model_name}"
        )

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
        Generate text from a prompt using a Hugging Face Inference API model.

        Parameters are accepted to mirror other clients; only max_output_tokens
        may be forwarded depending on the model.
        """
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        response = requests.post(self._endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Most text-generation models return a list with generated_text
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        return str(data)

