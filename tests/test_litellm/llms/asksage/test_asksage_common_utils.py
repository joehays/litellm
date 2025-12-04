"""
Unit tests for AskSage common utilities

Tests error handling, token caching, and token fetching logic.
All tests are mocked - no real script execution or API calls.
"""
import os
import subprocess
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from litellm.llms.asksage.common_utils import (
    AskSageError,
    TokenCache,
    get_asksage_token,
)


class TestAskSageError:
    """Test suite for AskSageError exception class"""

    def test_error_initialization(self):
        """Test basic error initialization"""
        error = AskSageError(status_code=401, message="Unauthorized")

        assert error.status_code == 401
        assert error.message == "Unauthorized"
        assert error.headers == {}

    def test_error_with_headers(self):
        """Test error initialization with headers"""
        headers = {"Content-Type": "application/json", "X-Request-ID": "123"}
        error = AskSageError(
            status_code=500, message="Internal Server Error", headers=headers
        )

        assert error.status_code == 500
        assert error.message == "Internal Server Error"
        assert error.headers == headers

    def test_error_inheritance(self):
        """Test that AskSageError properly inherits from BaseLLMException"""
        error = AskSageError(status_code=429, message="Rate limit exceeded")

        # Should have attributes from base exception
        assert hasattr(error, "status_code")
        assert hasattr(error, "message")
        assert hasattr(error, "headers")


class TestTokenCache:
    """Test suite for TokenCache class"""

    def test_cache_initialization(self):
        """Test cache initialization with default TTL"""
        cache = TokenCache()

        assert cache.token is None
        assert cache.timestamp == 0.0
        assert cache.ttl_seconds == 300  # Default 5 minutes

    def test_cache_initialization_custom_ttl(self):
        """Test cache initialization with custom TTL"""
        cache = TokenCache(ttl_seconds=600)

        assert cache.ttl_seconds == 600

    def test_cache_set_and_get(self):
        """Test setting and getting cached token"""
        cache = TokenCache()
        cache.set("test-token-12345")

        token = cache.get()
        assert token == "test-token-12345"

    def test_cache_expiry(self):
        """Test that cached token expires after TTL"""
        cache = TokenCache(ttl_seconds=1)  # 1 second TTL
        cache.set("test-token")

        # Should be valid immediately
        assert cache.get() == "test-token"

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert cache.get() is None

    def test_cache_clear(self):
        """Test clearing the cache"""
        cache = TokenCache()
        cache.set("test-token")

        assert cache.get() == "test-token"

        cache.clear()
        assert cache.token is None
        assert cache.timestamp == 0.0
        assert cache.get() is None

    def test_cache_get_without_set(self):
        """Test getting from empty cache"""
        cache = TokenCache()
        assert cache.get() is None


class TestGetAskSageToken:
    """Test suite for get_asksage_token function"""

    def test_token_from_cache(self):
        """Test that cached token is returned"""
        with patch("litellm.llms.asksage.common_utils._token_cache") as mock_cache:
            mock_cache.get.return_value = "cached-token-12345"

            token = get_asksage_token()

            assert token == "cached-token-12345"
            mock_cache.get.assert_called_once()

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_success(self, mock_run, mock_cache):
        """Test successful token fetch via command"""
        # Mock cache miss
        mock_cache.get.return_value = None

        # Mock successful command execution
        mock_result = Mock()
        mock_result.stdout = "command-token-67890\n"  # With trailing newline
        mock_run.return_value = mock_result

        with patch.dict(os.environ, {"ASKSAGE_TOKEN_COMMAND": "/path/to/script.sh"}):
            token = get_asksage_token()

        assert token == "command-token-67890"  # Should be stripped
        mock_cache.set.assert_called_once_with("command-token-67890")

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_with_tilde_expansion(self, mock_run, mock_cache):
        """Test that ~ is expanded in command path"""
        mock_cache.get.return_value = None

        mock_result = Mock()
        mock_result.stdout = "token-from-home\n"
        mock_run.return_value = mock_result

        with patch.dict(
            os.environ, {"ASKSAGE_TOKEN_COMMAND": "~/scripts/get-token.sh"}
        ):
            token = get_asksage_token()

        # Verify subprocess.run was called with expanded path
        call_args = mock_run.call_args[0][0]
        assert "~" not in call_args[0]  # Path should be expanded

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_timeout(self, mock_run, mock_cache):
        """Test handling of command timeout"""
        mock_cache.get.return_value = None
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="script", timeout=10)

        with patch.dict(
            os.environ,
            {
                "ASKSAGE_TOKEN_COMMAND": "/path/to/slow-script.sh",
                "ASKSAGE_API_KEY": "fallback-token",
            },
        ):
            token = get_asksage_token()

        # Should fall back to static token
        assert token == "fallback-token"

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_error(self, mock_run, mock_cache):
        """Test handling of command execution error"""
        mock_cache.get.return_value = None
        mock_run.side_effect = subprocess.CalledProcessError(1, "script")

        with patch.dict(
            os.environ,
            {
                "ASKSAGE_TOKEN_COMMAND": "/path/to/failing-script.sh",
                "ASKSAGE_API_KEY": "fallback-token",
            },
        ):
            token = get_asksage_token()

        # Should fall back to static token
        assert token == "fallback-token"

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_exception(self, mock_run, mock_cache):
        """Test handling of unexpected exception"""
        mock_cache.get.return_value = None
        mock_run.side_effect = Exception("Unexpected error")

        with patch.dict(
            os.environ,
            {
                "ASKSAGE_TOKEN_COMMAND": "/path/to/script.sh",
                "ASKSAGE_API_KEY": "fallback-token",
            },
        ):
            token = get_asksage_token()

        # Should fall back to static token
        assert token == "fallback-token"

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_from_command_empty_output(self, mock_run, mock_cache):
        """Test handling of command with empty output"""
        mock_cache.get.return_value = None

        mock_result = Mock()
        mock_result.stdout = "   \n  "  # Only whitespace
        mock_run.return_value = mock_result

        with patch.dict(
            os.environ,
            {
                "ASKSAGE_TOKEN_COMMAND": "/path/to/script.sh",
                "ASKSAGE_API_KEY": "fallback-token",
            },
        ):
            token = get_asksage_token()

        # Should fall back to static token since command output is empty
        assert token == "fallback-token"

    @patch("litellm.llms.asksage.common_utils._token_cache")
    def test_token_from_static_api_key(self, mock_cache):
        """Test token from ASKSAGE_API_KEY environment variable"""
        mock_cache.get.return_value = None

        with patch.dict(
            os.environ, {"ASKSAGE_API_KEY": "static-token-99999"}, clear=True
        ):
            token = get_asksage_token()

        assert token == "static-token-99999"
        mock_cache.set.assert_called_once_with("static-token-99999")

    @patch("litellm.llms.asksage.common_utils._token_cache")
    def test_token_no_source_available(self, mock_cache):
        """Test when no token source is available"""
        mock_cache.get.return_value = None

        with patch.dict(os.environ, {}, clear=True):
            token = get_asksage_token()

        assert token is None

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_command_priority_over_static(self, mock_run, mock_cache):
        """Test that token command has priority over static API key"""
        mock_cache.get.return_value = None

        mock_result = Mock()
        mock_result.stdout = "command-token"
        mock_run.return_value = mock_result

        with patch.dict(
            os.environ,
            {
                "ASKSAGE_TOKEN_COMMAND": "/path/to/script.sh",
                "ASKSAGE_API_KEY": "static-token",
            },
        ):
            token = get_asksage_token()

        # Should use command token, not static
        assert token == "command-token"

    @patch("litellm.llms.asksage.common_utils._token_cache")
    @patch("litellm.llms.asksage.common_utils.subprocess.run")
    def test_token_caching_from_command(self, mock_run, mock_cache):
        """Test that tokens from command are cached"""
        mock_cache.get.return_value = None

        mock_result = Mock()
        mock_result.stdout = "fresh-token"
        mock_run.return_value = mock_result

        with patch.dict(os.environ, {"ASKSAGE_TOKEN_COMMAND": "/path/to/script.sh"}):
            token = get_asksage_token()

        assert token == "fresh-token"
        mock_cache.set.assert_called_once_with("fresh-token")

    @patch("litellm.llms.asksage.common_utils._token_cache")
    def test_token_caching_from_static(self, mock_cache):
        """Test that static tokens are also cached"""
        mock_cache.get.return_value = None

        with patch.dict(os.environ, {"ASKSAGE_API_KEY": "static-token"}):
            token = get_asksage_token()

        assert token == "static-token"
        mock_cache.set.assert_called_once_with("static-token")
