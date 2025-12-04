"""
Unit tests for AskSage chat transformation logic

Tests request/response transformation between LiteLLM and AskSage API formats.
All tests are mocked - no real API calls.
"""
import json
import os
import sys
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.asksage.chat.transformation import AskSageConfig
from litellm.llms.asksage.common_utils import AskSageError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class TestAskSageConfig:
    """Test suite for AskSageConfig class"""

    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns 'asksage'"""
        config = AskSageConfig()
        assert config.custom_llm_provider == "asksage"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI parameters are returned"""
        config = AskSageConfig()
        supported = config.get_supported_openai_params("google-claude-4-opus")

        assert "temperature" in supported
        assert "max_tokens" in supported
        assert "stream" in supported

    def test_map_openai_params_temperature(self):
        """Test that temperature parameter is properly mapped"""
        config = AskSageConfig()
        optional_params = {}

        result = config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params=optional_params,
            model="google-claude-4-opus",
            drop_params=False,
        )

        assert result.get("temperature") == 0.7

    def test_map_openai_params_max_tokens(self):
        """Test that max_tokens parameter is properly mapped"""
        config = AskSageConfig()
        optional_params = {}

        result = config.map_openai_params(
            non_default_params={"max_tokens": 1000},
            optional_params=optional_params,
            model="google-claude-4-opus",
            drop_params=False,
        )

        assert result.get("max_tokens") == 1000

    def test_map_openai_params_multiple_params(self):
        """Test that multiple parameters are properly mapped"""
        config = AskSageConfig()
        optional_params = {}

        result = config.map_openai_params(
            non_default_params={"temperature": 0.8, "max_tokens": 2000},
            optional_params=optional_params,
            model="google-claude-4-opus",
            drop_params=False,
        )

        assert result.get("temperature") == 0.8
        assert result.get("max_tokens") == 2000


class TestAskSageValidateEnvironment:
    """Test suite for AskSage environment validation"""

    def test_validate_environment_with_api_key(self):
        """Test that API key is set in x-access-tokens header"""
        config = AskSageConfig()
        headers = {}

        result = config.validate_environment(
            headers=headers,
            model="google-claude-4-opus",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            api_key="test-token-12345",
            api_base="https://test.example.com",
        )

        assert result.get("x-access-tokens") == "test-token-12345"
        assert result.get("Content-Type") == "application/json"

    def test_validate_environment_without_api_key(self):
        """Test that headers are still set without API key"""
        config = AskSageConfig()
        headers = {}

        result = config.validate_environment(
            headers=headers,
            model="google-claude-4-opus",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            api_key=None,
            api_base="https://test.example.com",
        )

        assert "x-access-tokens" not in result
        assert result.get("Content-Type") == "application/json"

    def test_validate_environment_preserves_existing_headers(self):
        """Test that existing headers are preserved"""
        config = AskSageConfig()
        headers = {"Custom-Header": "custom-value"}

        result = config.validate_environment(
            headers=headers,
            model="google-claude-4-opus",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            api_key="test-token",
            api_base="https://test.example.com",
        )

        assert result.get("Custom-Header") == "custom-value"
        assert result.get("x-access-tokens") == "test-token"
        assert result.get("Content-Type") == "application/json"


class TestAskSageTransformRequest:
    """Test suite for AskSage request transformation"""

    def test_transform_request_simple_user_message(self):
        """Test transformation of simple user message"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "What is 2+2?"
        assert result["model"] == "google-claude-4-opus"
        assert result["dataset"] == ["none"]

    def test_transform_request_with_system_message(self):
        """Test transformation with system message"""
        config = AskSageConfig()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Hello!"
        assert result["system_prompt"] == "You are a helpful assistant."

    def test_transform_request_multiple_system_messages(self):
        """Test transformation with multiple system messages"""
        config = AskSageConfig()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You always respond concisely."},
            {"role": "user", "content": "Hello!"},
        ]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Hello!"
        assert "You are a helpful assistant." in result["system_prompt"]
        assert "You always respond concisely." in result["system_prompt"]

    def test_transform_request_with_dataset(self):
        """Test transformation with dataset parameter"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "What's in the dataset?"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"dataset": ["ds-12345"]},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "What's in the dataset?"
        assert result["dataset"] == ["ds-12345"]

    def test_transform_request_with_persona(self):
        """Test transformation with persona parameter"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "Help me code"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"persona": "coding_assistant"},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Help me code"
        assert result["persona"] == "coding_assistant"

    def test_transform_request_with_temperature(self):
        """Test transformation with temperature parameter"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "Write a poem"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"temperature": 0.9},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Write a poem"
        assert result["temperature"] == 0.9

    def test_transform_request_with_max_tokens(self):
        """Test transformation with max_tokens parameter"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "Tell me a story"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"max_tokens": 2000},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Tell me a story"
        assert result["max_tokens"] == 2000

    def test_transform_request_with_limit_references(self):
        """Test transformation with limit_references parameter"""
        config = AskSageConfig()
        messages = [{"role": "user", "content": "Find relevant papers"}]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"limit_references": 5},
            litellm_params={},
            headers={},
        )

        assert result["message"] == "Find relevant papers"
        assert result["limit_references"] == 5

    def test_transform_request_full_features(self):
        """Test transformation with all features combined"""
        config = AskSageConfig()
        messages = [
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "What does the literature say?"},
        ]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={
                "dataset": ["ds-papers-2024"],
                "persona": "researcher",
                "temperature": 0.7,
                "max_tokens": 1500,
                "limit_references": 10,
            },
            litellm_params={},
            headers={},
        )

        assert result["message"] == "What does the literature say?"
        assert result["system_prompt"] == "You are a researcher."
        assert result["dataset"] == ["ds-papers-2024"]
        assert result["persona"] == "researcher"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1500
        assert result["limit_references"] == 10

    def test_transform_request_system_prompt_override(self):
        """Test that explicit system_prompt overrides extracted system messages"""
        config = AskSageConfig()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = config.transform_request(
            model="google-claude-4-opus",
            messages=cast(list[AllMessageValues], messages),
            optional_params={"system_prompt": "Custom system prompt"},
            litellm_params={},
            headers={},
        )

        assert result["system_prompt"] == "Custom system prompt"


class TestAskSageTransformResponse:
    """Test suite for AskSage response transformation"""

    def test_transform_response_basic(self):
        """Test basic response transformation"""
        config = AskSageConfig()

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "The answer is 4.",
            "model_used": "google-claude-4-opus",
            "tokens_used": {"prompt": 10, "completion": 5, "total": 15},
        }
        mock_response.status_code = 200

        # Mock logging object
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        model_response = ModelResponse()

        result = config.transform_response(
            model="google-claude-4-opus",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "What is 2+2?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.choices) == 1
        assert result.choices[0].message.content == "The answer is 4."
        assert result.choices[0].message.role == "assistant"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.model == "google-claude-4-opus"

    def test_transform_response_with_citations(self):
        """Test response transformation with citations"""
        config = AskSageConfig()

        # Mock response with citations
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "According to the research...",
            "model_used": "google-claude-4-opus",
            "tokens_used": {"prompt": 20, "completion": 30, "total": 50},
            "citations": [
                {"title": "Paper 1", "url": "https://example.com/paper1"},
                {"title": "Paper 2", "url": "https://example.com/paper2"},
            ],
        }
        mock_response.status_code = 200

        # Mock logging object
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        model_response = ModelResponse()

        result = config.transform_response(
            model="google-claude-4-opus",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "What does the research say?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "According to the research..."
        assert hasattr(result, "_hidden_params")
        assert "citations" in result._hidden_params
        assert len(result._hidden_params["citations"]) == 2

    def test_transform_response_missing_fields(self):
        """Test response transformation with missing optional fields"""
        config = AskSageConfig()

        # Mock response with minimal fields
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response text",
            # No model_used, no tokens_used
        }
        mock_response.status_code = 200

        # Mock logging object
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        model_response = ModelResponse()

        result = config.transform_response(
            model="google-claude-4-opus",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Response text"
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_transform_response_invalid_json(self):
        """Test response transformation with invalid JSON"""
        config = AskSageConfig()

        # Mock response that raises exception on .json()
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_response.status_code = 200

        # Mock logging object
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        model_response = ModelResponse()

        with pytest.raises(AskSageError) as exc_info:
            config.transform_response(
                model="google-claude-4-opus",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging_obj,
                request_data={},
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert exc_info.value.status_code == 200
        assert "Failed to parse" in exc_info.value.message


class TestAskSageErrorHandling:
    """Test suite for AskSage error handling"""

    def test_get_error_class(self):
        """Test that error class is properly created"""
        config = AskSageConfig()

        error = config.get_error_class(
            error_message="Authentication failed",
            status_code=401,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, AskSageError)
        assert error.status_code == 401
        assert error.message == "Authentication failed"

    def test_get_error_class_with_httpx_headers(self):
        """Test error class creation with httpx.Headers"""
        config = AskSageConfig()

        httpx_headers = httpx.Headers({"Content-Type": "application/json"})
        error = config.get_error_class(
            error_message="Server error", status_code=500, headers=httpx_headers
        )

        assert isinstance(error, AskSageError)
        assert error.status_code == 500
        assert error.message == "Server error"
        assert isinstance(error.headers, dict)
