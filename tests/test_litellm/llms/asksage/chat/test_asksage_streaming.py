"""
Unit tests for AskSage Anthropic streaming (S23)

Tests the streaming functionality for Anthropic models via /server/anthropic/messages.
All tests are mocked - no real API calls.
"""
import json
import os
import sys
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.asksage.chat.handler import (
    AskSageAnthropicStreamIterator,
    AskSageChatCompletion,
    _is_anthropic_model,
)
from litellm.llms.asksage.common_utils import AskSageError


class TestIsAnthropicModel:
    """Test suite for _is_anthropic_model() helper function"""

    def test_recognizes_claude_3_sonnet(self):
        """Test that claude-3-sonnet is recognized as Anthropic"""
        assert _is_anthropic_model("claude-3-sonnet-20240229") is True

    def test_recognizes_claude_3_opus(self):
        """Test that claude-3-opus is recognized as Anthropic"""
        assert _is_anthropic_model("claude-3-opus-20240229") is True

    def test_recognizes_claude_4_opus(self):
        """Test that claude-4-opus is recognized as Anthropic"""
        assert _is_anthropic_model("claude-4-opus") is True

    def test_recognizes_asksage_claude_prefix(self):
        """Test that asksage/claude models are recognized"""
        assert _is_anthropic_model("asksage/claude-3-sonnet") is True

    def test_recognizes_anthropic_claude_prefix(self):
        """Test that anthropic/claude models are recognized"""
        assert _is_anthropic_model("anthropic/claude-3-haiku") is True

    def test_recognizes_claude_with_underscore(self):
        """Test that claude_ prefix is recognized"""
        assert _is_anthropic_model("claude_3_5_sonnet") is True

    def test_case_insensitive_detection(self):
        """Test that model detection is case insensitive"""
        assert _is_anthropic_model("CLAUDE-3-OPUS") is True
        assert _is_anthropic_model("Claude-3-Sonnet") is True

    def test_rejects_gpt_models(self):
        """Test that GPT models are not recognized as Anthropic"""
        assert _is_anthropic_model("gpt-4") is False
        assert _is_anthropic_model("gpt-4-turbo") is False
        assert _is_anthropic_model("gpt-3.5-turbo") is False

    def test_rejects_google_models(self):
        """Test that Google models are not recognized as Anthropic"""
        assert _is_anthropic_model("google-gemini-pro") is False
        assert _is_anthropic_model("gemini-pro") is False

    def test_rejects_llama_models(self):
        """Test that Llama models are not recognized as Anthropic"""
        assert _is_anthropic_model("llama-2-70b") is False
        assert _is_anthropic_model("meta/llama-3") is False

    def test_rejects_asksage_non_claude(self):
        """Test that asksage non-claude models are not recognized"""
        assert _is_anthropic_model("asksage/gpt-4") is False
        assert _is_anthropic_model("asksage/gemini-pro") is False

    def test_disable_streaming_env_var_1(self):
        """Test ASKSAGE_DISABLE_ANTHROPIC_STREAMING=1 disables streaming"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": "1"}):
            assert _is_anthropic_model("claude-3-sonnet-20240229") is False

    def test_disable_streaming_env_var_true(self):
        """Test ASKSAGE_DISABLE_ANTHROPIC_STREAMING=true disables streaming"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": "true"}):
            assert _is_anthropic_model("claude-3-opus") is False

    def test_disable_streaming_env_var_yes(self):
        """Test ASKSAGE_DISABLE_ANTHROPIC_STREAMING=yes disables streaming"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": "yes"}):
            assert _is_anthropic_model("claude-4-opus") is False

    def test_disable_streaming_env_var_case_insensitive(self):
        """Test ASKSAGE_DISABLE_ANTHROPIC_STREAMING is case insensitive"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": "TRUE"}):
            assert _is_anthropic_model("claude-3-sonnet") is False

    def test_disable_streaming_env_var_0_does_not_disable(self):
        """Test ASKSAGE_DISABLE_ANTHROPIC_STREAMING=0 does NOT disable streaming"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": "0"}):
            assert _is_anthropic_model("claude-3-sonnet-20240229") is True

    def test_disable_streaming_env_var_empty_does_not_disable(self):
        """Test empty ASKSAGE_DISABLE_ANTHROPIC_STREAMING does NOT disable"""
        with patch.dict(os.environ, {"ASKSAGE_DISABLE_ANTHROPIC_STREAMING": ""}):
            assert _is_anthropic_model("claude-3-sonnet-20240229") is True

    def test_streaming_enabled_when_env_var_not_set(self):
        """Test streaming is enabled when env var is not set"""
        env = os.environ.copy()
        env.pop("ASKSAGE_DISABLE_ANTHROPIC_STREAMING", None)
        with patch.dict(os.environ, env, clear=True):
            assert _is_anthropic_model("claude-3-sonnet-20240229") is True


class TestAskSageAnthropicStreamIterator:
    """Test suite for AskSageAnthropicStreamIterator"""

    def test_parse_sse_line_with_data(self):
        """Test parsing a valid SSE data line"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        chunk = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
        line = f"data: {json.dumps(chunk)}"
        result = iterator._parse_sse_line(line)

        assert result == chunk

    def test_parse_sse_line_done_marker(self):
        """Test parsing [DONE] marker"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        result = iterator._parse_sse_line("data: [DONE]")
        assert result == {"type": "done"}

    def test_parse_sse_line_empty(self):
        """Test parsing empty line returns None"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        assert iterator._parse_sse_line("") is None

    def test_parse_sse_line_comment(self):
        """Test parsing SSE comment returns None"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        assert iterator._parse_sse_line(":keepalive") is None

    def test_parse_sse_line_invalid_json(self):
        """Test parsing invalid JSON returns None"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        result = iterator._parse_sse_line("data: {invalid json}")
        assert result is None

    def test_chunk_to_response_content_block_delta(self):
        """Test converting content_block_delta to response"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "World"},
        }
        response = iterator._chunk_to_response(chunk)

        assert response.choices[0].delta.content == "World"
        assert response.model == "claude-3-sonnet"

    def test_chunk_to_response_message_start(self):
        """Test converting message_start with usage to response"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        chunk = {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "model": "claude-3-sonnet",
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        }
        response = iterator._chunk_to_response(chunk)

        assert response.usage.prompt_tokens == 100

    def test_chunk_to_response_message_delta(self):
        """Test converting message_delta with stop_reason to response"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        }
        response = iterator._chunk_to_response(chunk)

        assert response.choices[0].finish_reason == "stop"
        assert response.usage.completion_tokens == 50

    def test_map_finish_reason_end_turn(self):
        """Test mapping end_turn to stop"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        assert iterator._map_finish_reason("end_turn") == "stop"

    def test_map_finish_reason_max_tokens(self):
        """Test mapping max_tokens to length"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        assert iterator._map_finish_reason("max_tokens") == "length"

    def test_map_finish_reason_tool_use(self):
        """Test mapping tool_use to tool_calls"""
        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter([]),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        assert iterator._map_finish_reason("tool_use") == "tool_calls"

    def test_sync_iteration(self):
        """Test synchronous iteration over SSE stream"""
        lines = [
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
            "data: " + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}),
        ]

        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter(lines),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        results = list(iterator)
        assert len(results) == 2
        assert results[0].choices[0].delta.content == "Hi"
        assert results[1].choices[0].finish_reason == "stop"


class TestGetAnthropicBaseUrl:
    """Test suite for _get_anthropic_base_url method

    CAPRA's Anthropic streaming endpoint is at /server/anthropic/v1/messages
    (not /server/anthropic/messages). This is the full path required for
    direct HTTP calls since LiteLLM doesn't use the Anthropic SDK.
    """

    def test_converts_server_query_url(self):
        """Test conversion from /server/query to /server/anthropic/v1/messages"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.example.com/server/query")
        assert url == "https://api.example.com/server/anthropic/v1/messages"

    def test_handles_trailing_slash(self):
        """Test URL with trailing slash"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.example.com/server/query/")
        assert url == "https://api.example.com/server/anthropic/v1/messages"

    def test_handles_base_url_without_server_query(self):
        """Test URL without /server/query suffix"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.example.com")
        assert url == "https://api.example.com/server/anthropic/v1/messages"

    def test_handles_capra_flankspeed_url(self):
        """Test CAPRA flankspeed production URL"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.capra.flankspeed.us.navy.mil/server/query")
        assert url == "https://api.capra.flankspeed.us.navy.mil/server/anthropic/v1/messages"

    def test_strips_v1_suffix_before_constructing(self):
        """Test that /v1 suffix is stripped before constructing anthropic URL"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.example.com/v1")
        assert url == "https://api.example.com/server/anthropic/v1/messages"

    def test_strips_server_v1_suffix(self):
        """Test that /server/v1 suffix is stripped"""
        handler = AskSageChatCompletion()

        url = handler._get_anthropic_base_url("https://api.example.com/server/v1")
        assert url == "https://api.example.com/server/anthropic/v1/messages"


class TestTransformToAnthropicFormat:
    """Test suite for _transform_to_anthropic_format method"""

    def test_basic_user_message(self):
        """Test transformation of basic user message"""
        handler = AskSageChatCompletion()

        messages = [{"role": "user", "content": "Hello"}]
        result = handler._transform_to_anthropic_format(
            model="claude-3-sonnet",
            messages=messages,
            optional_params={},
        )

        assert result["model"] == "claude-3-sonnet"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["stream"] is True
        assert result["max_tokens"] == 1024  # Default

    def test_extracts_system_message(self):
        """Test that system message is extracted to system parameter"""
        handler = AskSageChatCompletion()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = handler._transform_to_anthropic_format(
            model="claude-3-sonnet",
            messages=messages,
            optional_params={},
        )

        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_combines_multiple_system_messages(self):
        """Test that multiple system messages are combined"""
        handler = AskSageChatCompletion()

        messages = [
            {"role": "system", "content": "First instruction."},
            {"role": "system", "content": "Second instruction."},
            {"role": "user", "content": "Hello"},
        ]
        result = handler._transform_to_anthropic_format(
            model="claude-3-sonnet",
            messages=messages,
            optional_params={},
        )

        assert result["system"] == "First instruction.\nSecond instruction."

    def test_includes_optional_params(self):
        """Test that optional parameters are included"""
        handler = AskSageChatCompletion()

        messages = [{"role": "user", "content": "Hello"}]
        result = handler._transform_to_anthropic_format(
            model="claude-3-sonnet",
            messages=messages,
            optional_params={"max_tokens": 500, "temperature": 0.7},
        )

        assert result["max_tokens"] == 500
        assert result["temperature"] == 0.7


class TestCompletionStreamRouting:
    """Test suite for completion method stream routing"""

    def _create_mock_logging_obj(self):
        """Create a mock logging object"""
        logging_obj = Mock()
        logging_obj.pre_call = Mock()
        return logging_obj

    def _create_mock_model_response(self):
        """Create a mock ModelResponse"""
        from litellm.types.utils import ModelResponse
        return ModelResponse()

    def test_routes_claude_stream_to_anthropic_endpoint(self):
        """Test that Claude models with stream=True route to Anthropic endpoint"""
        handler = AskSageChatCompletion()

        mock_stream_wrapper = Mock()

        with patch.object(handler, "_stream_anthropic_sync", return_value=mock_stream_wrapper) as mock_stream:
            result = handler.completion(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com/server/query",
                custom_llm_provider="asksage",
                custom_prompt_dict={},
                model_response=self._create_mock_model_response(),
                print_verbose=lambda x: None,
                encoding=None,
                api_key="test-token",
                logging_obj=self._create_mock_logging_obj(),
                optional_params={"stream": True},
                timeout=300.0,
                litellm_params={},
            )

            mock_stream.assert_called_once()
            assert result == mock_stream_wrapper

    def test_non_claude_stream_uses_standard_path(self):
        """Test that non-Claude models with stream=True use standard path"""
        import litellm.llms.asksage.chat.handler as handler_module

        # Temporarily set ASKSAGECLIENT_AVAILABLE to True for mocking
        original_value = handler_module.ASKSAGECLIENT_AVAILABLE
        try:
            handler_module.ASKSAGECLIENT_AVAILABLE = True
            handler = AskSageChatCompletion()

            mock_response_dict = {
                "message": "Response",
                "model_used": "gpt-4",
                "status": 200,
            }

            with patch.object(handler, "_stream_anthropic_sync") as mock_stream:
                with patch.object(handler, "_query_via_asksageclient", return_value=mock_response_dict):
                    handler.completion(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}],
                        api_base="https://api.example.com/server/query",
                        custom_llm_provider="asksage",
                        custom_prompt_dict={},
                        model_response=self._create_mock_model_response(),
                        print_verbose=lambda x: None,
                        encoding=None,
                        api_key="test-token",
                        logging_obj=self._create_mock_logging_obj(),
                        optional_params={"stream": True},
                        timeout=300.0,
                        litellm_params={},
                    )

                mock_stream.assert_not_called()
        finally:
            handler_module.ASKSAGECLIENT_AVAILABLE = original_value

    def test_non_stream_claude_uses_standard_path(self):
        """Test that Claude models without stream use standard path"""
        import litellm.llms.asksage.chat.handler as handler_module

        # Temporarily set ASKSAGECLIENT_AVAILABLE to True for mocking
        original_value = handler_module.ASKSAGECLIENT_AVAILABLE
        try:
            handler_module.ASKSAGECLIENT_AVAILABLE = True
            handler = AskSageChatCompletion()

            mock_response_dict = {
                "message": "Response",
                "model_used": "claude-3-sonnet",
                "status": 200,
            }

            with patch.object(handler, "_stream_anthropic_sync") as mock_stream:
                with patch.object(handler, "_query_via_asksageclient", return_value=mock_response_dict):
                    handler.completion(
                        model="claude-3-sonnet-20240229",
                        messages=[{"role": "user", "content": "Hello"}],
                        api_base="https://api.example.com/server/query",
                        custom_llm_provider="asksage",
                        custom_prompt_dict={},
                        model_response=self._create_mock_model_response(),
                        print_verbose=lambda x: None,
                        encoding=None,
                        api_key="test-token",
                        logging_obj=self._create_mock_logging_obj(),
                        optional_params={"stream": False},
                        timeout=300.0,
                        litellm_params={},
                    )

                mock_stream.assert_not_called()
        finally:
            handler_module.ASKSAGECLIENT_AVAILABLE = original_value


class TestAsyncStreamRouting:
    """Test suite for async stream routing"""

    def _create_mock_logging_obj(self):
        """Create a mock logging object"""
        logging_obj = Mock()
        logging_obj.pre_call = Mock()
        return logging_obj

    def _create_mock_model_response(self):
        """Create a mock ModelResponse"""
        from litellm.types.utils import ModelResponse
        return ModelResponse()

    @pytest.mark.asyncio
    async def test_async_routes_claude_stream_to_anthropic_endpoint(self):
        """Test that async Claude streaming routes correctly"""
        handler = AskSageChatCompletion()

        mock_stream_wrapper = AsyncMock()

        with patch.object(handler, "_astream_anthropic", return_value=mock_stream_wrapper) as mock_stream:
            result = handler.completion(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com/server/query",
                custom_llm_provider="asksage",
                custom_prompt_dict={},
                model_response=self._create_mock_model_response(),
                print_verbose=lambda x: None,
                encoding=None,
                api_key="test-token",
                logging_obj=self._create_mock_logging_obj(),
                optional_params={"stream": True},
                timeout=300.0,
                litellm_params={},
                acompletion=True,
            )

            mock_stream.assert_called_once()


class TestStreamingErrorHandling:
    """Test suite for streaming error handling"""

    def test_stream_iterator_handles_decode_error(self):
        """Test that stream iterator handles byte decoding"""
        lines = [
            b"data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}).encode(),
        ]

        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter(lines),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        result = next(iterator)
        assert result.choices[0].delta.content == "Hi"

    def test_chunk_response_id_is_consistent(self):
        """Test that response ID is consistent across chunks"""
        lines = [
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}}),
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "B"}}),
        ]

        iterator = AskSageAnthropicStreamIterator(
            streaming_response=iter(lines),
            sync_stream=True,
            model="claude-3-sonnet",
        )

        results = list(iterator)
        assert len(results) == 2
        assert results[0].id == results[1].id  # Same response ID
