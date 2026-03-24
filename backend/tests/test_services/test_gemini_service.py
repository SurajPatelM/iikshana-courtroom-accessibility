"""
Unit tests for Gemini, Groq, and HuggingFace service integrations.
All tests use mocking — no real API calls are made.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# -------------------------------------------------------
# GeminiClient tests
# -------------------------------------------------------
class TestGeminiClient:

    def test_model_name_is_stored(self):
        """GeminiClient must store and return the configured model name."""
        with patch("src.services.gemini_service.genai.Client"):
            from src.services.gemini_service import GeminiClient
            client = GeminiClient(model_name="gemini-1.5-flash", api_key="fake-key")
            assert client.model_name == "gemini-1.5-flash"

    def test_generate_text_returns_string(self):
        """generate_text must return a non-empty string from mocked response."""
        with patch("src.services.gemini_service.genai.Client") as mock_genai:
            mock_part = MagicMock()
            mock_part.text = "Mocked response"
            mock_candidate = MagicMock()
            mock_candidate.content.parts = [mock_part]
            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]
            mock_genai.return_value.models.generate_content.return_value = mock_response

            from src.services.gemini_service import GeminiClient
            client = GeminiClient(model_name="gemini-1.5-flash", api_key="fake-key")
            result = client.generate_text("Translate this")
            assert isinstance(result, str)
            assert result == "Mocked response"

    def test_generate_text_empty_candidates(self):
        """generate_text must return empty string when no candidates."""
        with patch("src.services.gemini_service.genai.Client") as mock_genai:
            mock_response = MagicMock()
            mock_response.candidates = []
            mock_genai.return_value.models.generate_content.return_value = mock_response

            from src.services.gemini_service import GeminiClient
            client = GeminiClient(model_name="gemini-1.5-flash", api_key="fake-key")
            result = client.generate_text("Translate this")
            assert result == ""

    def test_uses_env_api_key_when_not_provided(self):
        """GeminiClient must fall back to GEMINI_API_KEY env variable."""
        with patch("src.services.gemini_service.genai.Client") as mock_genai:
            with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
                from src.services.gemini_service import GeminiClient
                client = GeminiClient(model_name="gemini-1.5-flash")
                mock_genai.assert_called_once()


# -------------------------------------------------------
# GroqClient tests
# -------------------------------------------------------
class TestGroqClient:

    def test_raises_without_api_key(self):
        """GroqClient must raise ValueError when no API key is set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove GROQ_API_KEY if present
            import os
            os.environ.pop("GROQ_API_KEY", None)
            from src.services.groq_service import GroqClient
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                GroqClient(model_name="llama3-8b-8192")

    def test_model_name_is_stored(self):
        """GroqClient must store and return the configured model name."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
            from src.services.groq_service import GroqClient
            client = GroqClient(model_name="llama3-8b-8192")
            assert client.model_name == "llama3-8b-8192"

    def test_generate_text_returns_string(self):
        """generate_text must return the content from mocked Groq response."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
            with patch("src.services.groq_service.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Translated text"}}]
                }
                mock_post.return_value = mock_response

                from src.services.groq_service import GroqClient
                client = GroqClient(model_name="llama3-8b-8192")
                result = client.generate_text("Translate this")
                assert result == "Translated text"

    def test_rate_limit_retry(self):
        """GroqClient must retry on 429 rate limit responses."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
            with patch("src.services.groq_service.requests.post") as mock_post:
                with patch("src.services.groq_service.time.sleep"):
                    rate_limit = MagicMock()
                    rate_limit.status_code = 429

                    success = MagicMock()
                    success.status_code = 200
                    success.json.return_value = {
                        "choices": [{"message": {"content": "Retried response"}}]
                    }
                    mock_post.side_effect = [rate_limit, rate_limit, success]

                    from src.services.groq_service import GroqClient
                    client = GroqClient(model_name="llama3-8b-8192")
                    result = client.generate_text("Translate this")
                    assert result == "Retried response"
                    assert mock_post.call_count == 3


# -------------------------------------------------------
# HuggingFaceClient tests
# -------------------------------------------------------
class TestHuggingFaceClient:

    def test_raises_without_api_token(self):
        """HuggingFaceClient must raise ValueError when no token is set."""
        import os
        os.environ.pop("HF_API_TOKEN", None)
        from src.services.hf_service import HuggingFaceClient
        with pytest.raises(ValueError, match="HF_API_TOKEN"):
            HuggingFaceClient(model_name="Helsinki-NLP/opus-mt-en-es")

    def test_model_name_is_stored(self):
        """HuggingFaceClient must store and return the configured model name."""
        with patch.dict("os.environ", {"HF_API_TOKEN": "fake-token"}):
            from src.services.hf_service import HuggingFaceClient
            client = HuggingFaceClient(model_name="Helsinki-NLP/opus-mt-en-es")
            assert client.model_name == "Helsinki-NLP/opus-mt-en-es"

    def test_generate_text_returns_string(self):
        """generate_text must return generated_text from mocked HF response."""
        with patch.dict("os.environ", {"HF_API_TOKEN": "fake-token"}):
            with patch("src.services.hf_service.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = [
                    {"generated_text": "Texto traducido"}
                ]
                mock_post.return_value = mock_response

                from src.services.hf_service import HuggingFaceClient
                client = HuggingFaceClient(model_name="Helsinki-NLP/opus-mt-en-es")
                result = client.generate_text("Translate this")
                assert result == "Texto traducido"

    def test_system_prompt_prepended(self):
        """System prompt must be prepended to user prompt for HF API."""
        with patch.dict("os.environ", {"HF_API_TOKEN": "fake-token"}):
            with patch("src.services.hf_service.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = [{"generated_text": "output"}]
                mock_post.return_value = mock_response

                from src.services.hf_service import HuggingFaceClient
                client = HuggingFaceClient(model_name="Helsinki-NLP/opus-mt-en-es")
                client.generate_text("user prompt", system_prompt="system prompt")

                call_body = mock_post.call_args[1]["json"]
                assert "system prompt" in call_body["inputs"]
                assert "user prompt" in call_body["inputs"]