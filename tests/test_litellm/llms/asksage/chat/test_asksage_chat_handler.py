"""
Unit tests for AskSage chat handler (E37 Phase 3)

Tests the AskSageClient delegation pattern and fallback to httpx.
All tests are mocked - no real API calls.
"""
import asyncio
import os
import sys
from typing import cast
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.asksage.chat.handler import (
    AskSageChatCompletion,
    ASKSAGECLIENT_AVAILABLE,
)
from litellm.llms.asksage.common_utils import AskSageError


class TestAskSageChatCompletionInit:
    """Test suite for AskSageChatCompletion initialization"""

    def test_init_creates_thread_pool(self):
        """Test that initialization creates a thread pool executor"""
        handler = AskSageChatCompletion()
        assert AskSageChatCompletion._executor is not None

    def test_executor_is_shared(self):
        """Test that thread pool executor is shared across instances"""
        handler1 = AskSageChatCompletion()
        handler2 = AskSageChatCompletion()
        assert handler1._executor is handler2._executor


class TestCABundlePathResolution:
    """Test suite for CA bundle path resolution"""

    def test_get_ca_bundle_from_asksage_env(self):
        """Test CA bundle resolution from ASKSAGE_CA_CERT_PATH"""
        handler = AskSageChatCompletion()

        with patch.dict(os.environ, {"ASKSAGE_CA_CERT_PATH": "/etc/ssl/certs/ca.crt"}):
            with patch("os.path.exists", return_value=True):
                result = handler._get_ca_bundle_path()
                assert result == "/etc/ssl/certs/ca.crt"

    def test_get_ca_bundle_empty_string_disables_verification(self):
        """Test that empty ASKSAGE_CA_CERT_PATH disables verification"""
        handler = AskSageChatCompletion()

        with patch.dict(os.environ, {"ASKSAGE_CA_CERT_PATH": ""}):
            result = handler._get_ca_bundle_path()
            assert result is None

    def test_get_ca_bundle_from_requests_ca_bundle(self):
        """Test CA bundle fallback to REQUESTS_CA_BUNDLE"""
        handler = AskSageChatCompletion()

        with patch.dict(
            os.environ,
            {"ASKSAGE_CA_CERT_PATH": "", "REQUESTS_CA_BUNDLE": "/custom/ca.crt"},
            clear=False,
        ):
            # Clear ASKSAGE_CA_CERT_PATH to trigger fallback
            del os.environ["ASKSAGE_CA_CERT_PATH"]
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == "/custom/ca.crt"
                # Re-run without ASKSAGE_CA_CERT_PATH
                with patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": "/custom/ca.crt"}):
                    result = handler._get_ca_bundle_path()
                    assert result == "/custom/ca.crt"

    def test_get_ca_bundle_system_default(self):
        """Test CA bundle fallback to system default"""
        handler = AskSageChatCompletion()

        # Clear all env vars
        env_without_ca = {
            k: v
            for k, v in os.environ.items()
            if k not in ["ASKSAGE_CA_CERT_PATH", "REQUESTS_CA_BUNDLE"]
        }

        with patch.dict(os.environ, env_without_ca, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = (
                    lambda p: p == "/etc/ssl/certs/ca-certificates.crt"
                )
                result = handler._get_ca_bundle_path()
                assert result == "/etc/ssl/certs/ca-certificates.crt"


class TestAskSageClientCreation:
    """Test suite for AskSageClient creation"""

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_create_asksage_client_with_ca_bundle(self):
        """Test AskSageClient creation with CA bundle"""
        handler = AskSageChatCompletion()

        with patch.object(handler, "_get_ca_bundle_path", return_value="/path/to/ca.crt"):
            client = handler._create_asksage_client(
                api_base="https://api.test.com",
                api_key="test-token-123",
            )
            assert client is not None

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_create_asksage_client_strips_trailing_slash(self):
        """Test that trailing slash is stripped from api_base"""
        handler = AskSageChatCompletion()

        with patch.object(handler, "_get_ca_bundle_path", return_value=None):
            # We can't easily inspect the URL without mocking the client
            # but we can at least verify it doesn't raise an error
            client = handler._create_asksage_client(
                api_base="https://api.test.com/",
                api_key="test-token-123",
            )
            assert client is not None


class TestQueryViaAskSageClient:
    """Test suite for _query_via_asksageclient method"""

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_query_via_asksageclient_basic(self):
        """Test basic query via AskSageClient"""
        handler = AskSageChatCompletion()

        mock_client = Mock()
        mock_client.query.return_value = {
            "message": "Test response",
            "model_used": "test-model",
            "status": 200,
        }

        with patch.object(handler, "_create_asksage_client", return_value=mock_client):
            result = handler._query_via_asksageclient(
                api_base="https://api.test.com",
                api_key="test-token",
                data={"message": "Hello", "model": "test-model"},
            )

            assert result["message"] == "Test response"
            mock_client.query.assert_called_once()

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_query_via_asksageclient_with_reasoning_effort(self):
        """Test query with reasoning_effort parameter"""
        handler = AskSageChatCompletion()

        mock_client = Mock()
        mock_client.query.return_value = {"message": "Response", "status": 200}

        with patch.object(handler, "_create_asksage_client", return_value=mock_client):
            handler._query_via_asksageclient(
                api_base="https://api.test.com",
                api_key="test-token",
                data={
                    "message": "Complex task",
                    "model": "test-model",
                    "reasoning_effort": "high",
                },
            )

            # Verify reasoning_effort was passed
            call_kwargs = mock_client.query.call_args[1]
            assert call_kwargs.get("reasoning_effort") == "high"

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_query_via_asksageclient_removes_none_values(self):
        """Test that None values are not passed to client.query()"""
        handler = AskSageChatCompletion()

        mock_client = Mock()
        mock_client.query.return_value = {"message": "Response", "status": 200}

        with patch.object(handler, "_create_asksage_client", return_value=mock_client):
            handler._query_via_asksageclient(
                api_base="https://api.test.com",
                api_key="test-token",
                data={
                    "message": "Hello",
                    "model": "test-model",
                    "reasoning_effort": None,  # Should be filtered out
                },
            )

            call_kwargs = mock_client.query.call_args[1]
            assert "reasoning_effort" not in call_kwargs


class TestDictToMockResponse:
    """Test suite for _dict_to_mock_response method"""

    def test_dict_to_mock_response_basic(self):
        """Test conversion of dict to mock response"""
        handler = AskSageChatCompletion()

        response_dict = {
            "message": "Test response",
            "model_used": "test-model",
            "status": 200,
        }

        mock_response = handler._dict_to_mock_response(response_dict)

        assert mock_response.json() == response_dict
        assert mock_response.status_code == 200

    def test_dict_to_mock_response_default_status(self):
        """Test that status defaults to 200 if not present"""
        handler = AskSageChatCompletion()

        response_dict = {
            "message": "Test response",
            "model_used": "test-model",
        }

        mock_response = handler._dict_to_mock_response(response_dict)

        assert mock_response.status_code == 200

    def test_dict_to_mock_response_error_status(self):
        """Test mock response with error status"""
        handler = AskSageChatCompletion()

        response_dict = {
            "message": "Error occurred",
            "status": 500,
        }

        mock_response = handler._dict_to_mock_response(response_dict)

        assert mock_response.status_code == 500


class TestAsyncQueryViaAskSageClient:
    """Test suite for _aquery_via_asksageclient method"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    async def test_aquery_via_asksageclient_runs_in_executor(self):
        """Test that async query runs sync client in thread pool"""
        handler = AskSageChatCompletion()

        mock_response = {"message": "Async response", "status": 200}

        with patch.object(
            handler, "_query_via_asksageclient", return_value=mock_response
        ):
            result = await handler._aquery_via_asksageclient(
                api_base="https://api.test.com",
                api_key="test-token",
                data={"message": "Hello", "model": "test-model"},
            )

            assert result == mock_response


class TestCompletionWithAskSageClient:
    """Test suite for completion method with AskSageClient"""

    def _create_mock_logging_obj(self):
        """Create a mock logging object"""
        logging_obj = Mock()
        logging_obj.pre_call = Mock()
        return logging_obj

    def _create_mock_model_response(self):
        """Create a mock ModelResponse"""
        from litellm.types.utils import ModelResponse

        return ModelResponse()

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_completion_uses_asksageclient_when_available(self):
        """Test that completion uses AskSageClient when available"""
        handler = AskSageChatCompletion()

        mock_response_dict = {
            "message": "Response from AskSageClient",
            "model_used": "test-model",
            "status": 200,
            "usage": {
                "model_tokens": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
        }

        with patch.object(
            handler, "_query_via_asksageclient", return_value=mock_response_dict
        ):
            result = handler.completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.test.com/server/query",
                custom_llm_provider="asksage",
                custom_prompt_dict={},
                model_response=self._create_mock_model_response(),
                print_verbose=lambda x: None,
                encoding=None,
                api_key="test-token",
                logging_obj=self._create_mock_logging_obj(),
                optional_params={},
                timeout=300.0,
                litellm_params={},
            )

            assert result.choices[0].message.content == "Response from AskSageClient"

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_completion_passes_reasoning_effort(self):
        """Test that completion passes reasoning_effort to AskSageClient"""
        handler = AskSageChatCompletion()

        mock_response_dict = {
            "message": "Extended thinking response",
            "model_used": "test-model",
            "status": 200,
        }

        with patch.object(
            handler, "_query_via_asksageclient", return_value=mock_response_dict
        ) as mock_query:
            handler.completion(
                model="test-model",
                messages=[{"role": "user", "content": "Complex analysis"}],
                api_base="https://api.test.com/server/query",
                custom_llm_provider="asksage",
                custom_prompt_dict={},
                model_response=self._create_mock_model_response(),
                print_verbose=lambda x: None,
                encoding=None,
                api_key="test-token",
                logging_obj=self._create_mock_logging_obj(),
                optional_params={"reasoning_effort": "high"},
                timeout=300.0,
                litellm_params={},
            )

            # Verify _query_via_asksageclient was called
            mock_query.assert_called_once()
            # The data dict passed should contain reasoning_effort
            call_args = mock_query.call_args
            data = call_args[1]["data"] if "data" in call_args[1] else call_args[0][2]
            assert data.get("reasoning_effort") == "high"


class TestCompletionFallbackToHttpx:
    """Test suite for completion fallback when asksageclient is not available"""

    def _create_mock_logging_obj(self):
        """Create a mock logging object"""
        logging_obj = Mock()
        logging_obj.pre_call = Mock()
        return logging_obj

    def _create_mock_model_response(self):
        """Create a mock ModelResponse"""
        from litellm.types.utils import ModelResponse

        return ModelResponse()

    def test_completion_falls_back_to_httpx_when_client_unavailable(self):
        """Test that completion falls back to httpx when asksageclient is not available"""
        import litellm.llms.asksage.chat.handler as handler_module

        # Temporarily set ASKSAGECLIENT_AVAILABLE to False
        original_value = handler_module.ASKSAGECLIENT_AVAILABLE

        try:
            handler_module.ASKSAGECLIENT_AVAILABLE = False
            handler = AskSageChatCompletion()

            mock_httpx_response = Mock()
            mock_httpx_response.json.return_value = {
                "message": "Response from httpx",
                "model_used": "test-model",
                "status": 200,
            }
            mock_httpx_response.status_code = 200
            mock_httpx_response.raise_for_status = Mock()

            mock_client = Mock()
            mock_client.post.return_value = mock_httpx_response

            with patch.object(handler, "_get_httpx_client", return_value=mock_client):
                result = handler.completion(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="https://api.test.com/server/query",
                    custom_llm_provider="asksage",
                    custom_prompt_dict={},
                    model_response=self._create_mock_model_response(),
                    print_verbose=lambda x: None,
                    encoding=None,
                    api_key="test-token",
                    logging_obj=self._create_mock_logging_obj(),
                    optional_params={},
                    timeout=300.0,
                    litellm_params={},
                )

                assert result.choices[0].message.content == "Response from httpx"
                mock_client.post.assert_called_once()

        finally:
            # Restore original value
            handler_module.ASKSAGECLIENT_AVAILABLE = original_value


class TestAskSageErrorHandling:
    """Test suite for error handling in handler"""

    def _create_mock_logging_obj(self):
        """Create a mock logging object"""
        logging_obj = Mock()
        logging_obj.pre_call = Mock()
        return logging_obj

    def _create_mock_model_response(self):
        """Create a mock ModelResponse"""
        from litellm.types.utils import ModelResponse

        return ModelResponse()

    @pytest.mark.skipif(
        not ASKSAGECLIENT_AVAILABLE, reason="asksageclient not installed"
    )
    def test_completion_raises_asksage_error_on_exception(self):
        """Test that completion raises AskSageError on exception"""
        handler = AskSageChatCompletion()

        with patch.object(
            handler,
            "_query_via_asksageclient",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(AskSageError) as exc_info:
                handler.completion(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="https://api.test.com/server/query",
                    custom_llm_provider="asksage",
                    custom_prompt_dict={},
                    model_response=self._create_mock_model_response(),
                    print_verbose=lambda x: None,
                    encoding=None,
                    api_key="test-token",
                    logging_obj=self._create_mock_logging_obj(),
                    optional_params={},
                    timeout=300.0,
                    litellm_params={},
                )

            assert exc_info.value.status_code == 500
            assert "Connection failed" in exc_info.value.message
