"""
Handler for AskSage chat completions

Implements the chat completion API for AskSage/CAPRA.

E37 Phase 3: Delegates to official asksageclient package for API calls.
This provides automatic support for:
- reasoning_effort parameter (Extended Thinking)
- tools and tool_choice (function calling)
- live mode (web search)
- All future AskSage API features

E37 Phase 4: Adds MARS extensions:
- Prometheus metrics (request count, latency, tokens)
- Structured logging with secret redaction
- Deprecation warnings for legacy httpx path

Reference: https://docs.asksage.ai/docs/api-documentation/ask-sage-python-client.html
"""
import asyncio
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import ModelResponse
from litellm.utils import ProviderConfigManager

from ...base import BaseLLM
from ..common_utils import AskSageError
from .transformation import AskSageConfig

# Set up structured logging
logger = logging.getLogger("litellm.asksage")

# Import Prometheus metrics (optional - graceful fallback if not available)
try:
    from prometheus_client import Counter, Histogram

    ASKSAGE_REQUESTS_TOTAL = Counter(
        "asksage_requests_total",
        "Total number of AskSage API requests",
        ["model", "status", "method"],
    )
    ASKSAGE_REQUEST_DURATION_SECONDS = Histogram(
        "asksage_request_duration_seconds",
        "AskSage API request latency in seconds",
        ["model", "method"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )
    ASKSAGE_TOKENS_TOTAL = Counter(
        "asksage_tokens_total",
        "Total tokens used in AskSage API calls",
        ["model", "type"],
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    ASKSAGE_REQUESTS_TOTAL = None
    ASKSAGE_REQUEST_DURATION_SECONDS = None
    ASKSAGE_TOKENS_TOTAL = None

# Import official AskSage client
try:
    from asksageclient import AskSageClient

    ASKSAGECLIENT_AVAILABLE = True
except ImportError:
    ASKSAGECLIENT_AVAILABLE = False
    AskSageClient = None  # type: ignore


def _redact_api_key(api_key: Optional[str], visible_chars: int = 8) -> str:
    """Redact API key for safe logging, showing only first N characters."""
    if not api_key:
        return "NONE"
    if len(api_key) <= visible_chars:
        return "*" * len(api_key)
    return f"{api_key[:visible_chars]}...{'*' * (len(api_key) - visible_chars)}"


def _record_request_metrics(
    model: str, status: str, method: str, duration: float, tokens: Optional[dict] = None
) -> None:
    """Record Prometheus metrics for an AskSage request."""
    if not PROMETHEUS_AVAILABLE:
        return

    # Record request count
    ASKSAGE_REQUESTS_TOTAL.labels(model=model, status=status, method=method).inc()

    # Record latency
    ASKSAGE_REQUEST_DURATION_SECONDS.labels(model=model, method=method).observe(duration)

    # Record tokens if available
    if tokens:
        if "prompt_tokens" in tokens:
            ASKSAGE_TOKENS_TOTAL.labels(model=model, type="prompt").inc(tokens["prompt_tokens"])
        if "completion_tokens" in tokens:
            ASKSAGE_TOKENS_TOTAL.labels(model=model, type="completion").inc(tokens["completion_tokens"])


class AskSageChatCompletion(BaseLLM):
    """
    Handler for AskSage chat completion API

    Supports both standard AskSage (https://api.asksage.ai) and
    CAPRA (https://api.capra.flankspeed.us.navy.mil) endpoints.

    E37 Phase 3: Uses official asksageclient package for API calls when available.
    Falls back to httpx for backward compatibility if asksageclient not installed.
    """

    # Thread pool for running sync asksageclient in async context
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(self) -> None:
        super().__init__()
        # Lazy initialization of thread pool
        if AskSageChatCompletion._executor is None:
            AskSageChatCompletion._executor = ThreadPoolExecutor(max_workers=4)

    def _get_ca_bundle_path(self) -> Optional[str]:
        """
        Get CA bundle path for TLS verification.

        Resolution order:
        1. ASKSAGE_CA_CERT_PATH environment variable
        2. REQUESTS_CA_BUNDLE environment variable
        3. System default (/etc/ssl/certs/ca-certificates.crt)

        Returns None if ASKSAGE_CA_CERT_PATH is set to empty string (disables verification).
        """
        ca_cert_path = os.environ.get("ASKSAGE_CA_CERT_PATH")
        if ca_cert_path == "":
            # Empty string = explicitly disable verification (INSECURE)
            return None
        if ca_cert_path and os.path.exists(ca_cert_path):
            return ca_cert_path

        # Fall back to REQUESTS_CA_BUNDLE
        requests_ca = os.environ.get("REQUESTS_CA_BUNDLE")
        if requests_ca and os.path.exists(requests_ca):
            return requests_ca

        # System default
        default_ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(default_ca):
            return default_ca

        return None

    def _create_asksage_client(self, api_base: str, api_key: str) -> "AskSageClient":
        """
        Create AskSageClient with proper SSL/CA bundle configuration.

        Args:
            api_base: Base URL for AskSage API (e.g., https://api.asksage.ai)
            api_key: Bearer token for authentication

        Returns:
            Configured AskSageClient instance
        """
        ca_bundle = self._get_ca_bundle_path()

        # AskSageClient uses email/api_key for auth
        # For token-based auth (CAPRA), email is empty and api_key is the bearer token
        return AskSageClient(
            email="",  # Empty for token-based auth
            api_key=api_key,
            server_base_url=api_base.rstrip("/"),  # Remove trailing slash
            path_to_CA_Bundle=ca_bundle,
        )

    def _get_httpx_client(
        self, api_base: str, timeout: Union[float, httpx.Timeout]
    ) -> httpx.Client:
        """
        Create httpx client with optional TLS certificate for CAPRA

        Checks for ASKSAGE_CA_CERT_PATH environment variable.
        If present, uses it for TLS verification (required for CAPRA/DoD endpoints).
        If set to empty string, disables verification (INSECURE - for development only).
        """
        ca_cert_path = os.environ.get("ASKSAGE_CA_CERT_PATH")

        if ca_cert_path == "":
            # Empty string = explicitly disable verification (INSECURE)
            return httpx.Client(timeout=timeout, verify=False)
        elif ca_cert_path and os.path.exists(ca_cert_path):
            # Use DoD CA certificate chain for CAPRA
            return httpx.Client(timeout=timeout, verify=ca_cert_path)
        else:
            # Standard HTTPS (use system CA bundle)
            return _get_httpx_client(params={"timeout": timeout})

    def _get_async_httpx_client(
        self, api_base: str, timeout: Union[float, httpx.Timeout]
    ) -> httpx.AsyncClient:
        """
        Create async httpx client with optional TLS certificate for CAPRA
        If ASKSAGE_CA_CERT_PATH is empty string, disables verification (INSECURE).
        """
        ca_cert_path = os.environ.get("ASKSAGE_CA_CERT_PATH")

        if ca_cert_path == "":
            # Empty string = explicitly disable verification (INSECURE)
            return httpx.AsyncClient(timeout=timeout, verify=False)
        elif ca_cert_path and os.path.exists(ca_cert_path):
            # Use DoD CA certificate chain for CAPRA
            return httpx.AsyncClient(timeout=timeout, verify=ca_cert_path)
        else:
            # Standard HTTPS (use system CA bundle)
            return get_async_httpx_client(
                llm_provider=litellm.LlmProviders.ASKSAGE, params={"timeout": timeout}
            )

    def _query_via_asksageclient(
        self, api_base: str, api_key: str, data: dict
    ) -> dict:
        """
        Execute query using official AskSageClient.

        Args:
            api_base: Base URL for AskSage API
            api_key: Bearer token for authentication
            data: Request data dict (from transform_request)

        Returns:
            AskSage API response dict
        """
        client = self._create_asksage_client(api_base, api_key)

        # Map data dict to AskSageClient.query() parameters
        # Note: AskSageClient.query() takes different parameter names
        query_params = {
            "message": data.get("message", ""),
            "model": data.get("model"),
            "persona": data.get("persona", "default"),
            "dataset": data.get("dataset", ["none"]),
            "temperature": data.get("temperature", 0.0),
            "system_prompt": data.get("system_prompt"),
            "limit_references": data.get("limit_references"),
            "live": data.get("live", 0),
            "tools": data.get("tools"),
            "tool_choice": data.get("tool_choice"),
            "reasoning_effort": data.get("reasoning_effort"),
        }

        # Remove None values to use client defaults
        query_params = {k: v for k, v in query_params.items() if v is not None}

        # Log request (structured logging with secret redaction)
        model = query_params.get("model", "unknown")
        logger.debug(
            "AskSageClient.query() request",
            extra={
                "params": list(query_params.keys()),
                "model": model,
                "reasoning_effort": query_params.get("reasoning_effort"),
                "has_tools": "tools" in query_params,
                "api_base": api_base,
            },
        )

        # Execute query with timing
        start_time = time.time()
        try:
            response = client.query(**query_params)
            duration = time.time() - start_time

            # Extract token usage for metrics
            tokens = {}
            if isinstance(response, dict):
                tokens = {
                    "prompt_tokens": response.get("prompt_tokens", 0),
                    "completion_tokens": response.get("completion_tokens", 0),
                }

            # Record metrics
            _record_request_metrics(
                model=model,
                status="success",
                method="asksageclient",
                duration=duration,
                tokens=tokens,
            )

            # Log response (without sensitive content)
            logger.debug(
                "AskSageClient.query() response",
                extra={
                    "model": model,
                    "duration_seconds": round(duration, 3),
                    "prompt_tokens": tokens.get("prompt_tokens"),
                    "completion_tokens": tokens.get("completion_tokens"),
                    "status": response.get("status") if isinstance(response, dict) else None,
                },
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            _record_request_metrics(
                model=model,
                status="error",
                method="asksageclient",
                duration=duration,
            )
            logger.error(
                "AskSageClient.query() failed",
                extra={
                    "model": model,
                    "duration_seconds": round(duration, 3),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise

    def _dict_to_mock_response(self, response_dict: dict) -> httpx.Response:
        """
        Convert AskSageClient response dict to httpx.Response-like object.

        This allows reuse of existing transform_response() logic.
        """
        # Create a mock response object that transform_response can use
        class MockResponse:
            def __init__(self, data: dict):
                self._data = data
                self.status_code = data.get("status", 200)

            def json(self) -> dict:
                return self._data

        return MockResponse(response_dict)  # type: ignore

    async def _aquery_via_asksageclient(
        self, api_base: str, api_key: str, data: dict
    ) -> dict:
        """
        Execute async query using official AskSageClient.

        Since AskSageClient is synchronous, we run it in a thread pool executor.
        """
        loop = asyncio.get_event_loop()

        def _sync_query():
            return self._query_via_asksageclient(api_base, api_key, data)

        return await loop.run_in_executor(self._executor, _sync_query)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion: bool = False,
        logger_fn=None,
        headers: Optional[dict] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        """
        Main completion handler for AskSage

        Args:
            model: Model name (e.g., "google-claude-4-opus")
            messages: List of message dicts
            api_base: Base URL for AskSage API
            api_key: Bearer token for authentication
            optional_params: Additional parameters (dataset, persona, etc.)
            acompletion: If True, use async completion
            headers: HTTP headers
            client: Optional HTTP client

        Returns:
            ModelResponse object
        """
        from litellm.utils import ProviderConfigManager

        # Structured logging with secret redaction (E37 Phase 4)
        logger.debug(
            "AskSage completion request",
            extra={
                "model": model,
                "api_base": api_base,
                "api_key": _redact_api_key(api_key),
                "message_count": len(messages),
                "acompletion": acompletion,
            },
        )

        headers = headers or {}

        # Validate environment and set up headers
        config = AskSageConfig()
        headers = config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Transform request to AskSage format
        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Log request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if acompletion is True:
            return self.acompletion(
                model=model,
                messages=messages,
                api_base=api_base,
                data=data,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                timeout=timeout,
                client=client,
                litellm_params=litellm_params,
            )

        # Synchronous completion - try AskSageClient first, fall back to httpx
        if ASKSAGECLIENT_AVAILABLE:
            # Use official AskSageClient (E37 Phase 3)
            logger.debug("Using AskSageClient for completion", extra={"model": model})
            try:
                response_dict = self._query_via_asksageclient(api_base, api_key, data)
                response = self._dict_to_mock_response(response_dict)
            except Exception as e:
                # Map asksageclient exceptions to AskSageError
                error_msg = str(e)
                status_code = 500
                # Try to extract status code from exception if available
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                raise AskSageError(status_code=status_code, message=error_msg)
        else:
            # Fall back to direct httpx (legacy path - DEPRECATED)
            # E37 Phase 6: This path is deprecated and will be removed in a future version
            warnings.warn(
                "Direct httpx fallback for AskSage is deprecated. "
                "Install asksageclient>=1.42 for full feature support: "
                "pip install asksageclient",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "AskSageClient not available, using deprecated httpx fallback",
                extra={"model": model, "api_base": api_base},
            )
            if client is None or not isinstance(client, HTTPHandler):
                client = self._get_httpx_client(api_base=api_base, timeout=timeout)

            start_time = time.time()
            try:
                response = client.post(
                    api_base, headers=headers, json=data, timeout=timeout
                )
                response.raise_for_status()
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="success", method="httpx", duration=duration
                )
            except httpx.HTTPStatusError as e:
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="error", method="httpx", duration=duration
                )
                error_headers = getattr(e, "headers", None)
                error_response = getattr(e, "response", None)
                if error_headers is None and error_response:
                    error_headers = getattr(error_response, "headers", None)

                raise AskSageError(
                    status_code=e.response.status_code,
                    message=e.response.text,
                    headers=error_headers,
                )
            except Exception as e:
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="error", method="httpx", duration=duration
                )
                raise AskSageError(status_code=500, message=str(e))

        # Transform response
        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        """
        Async completion handler for AskSage

        E37 Phase 3: Uses official AskSageClient when available.
        Since AskSageClient is synchronous, we run it in a thread pool executor.

        E37 Phase 4: Adds Prometheus metrics and structured logging.
        """
        # Async completion - try AskSageClient first, fall back to httpx
        if ASKSAGECLIENT_AVAILABLE:
            # Use official AskSageClient (E37 Phase 3)
            logger.debug(
                "Using AskSageClient for async completion (via thread pool)",
                extra={"model": model},
            )
            try:
                response_dict = await self._aquery_via_asksageclient(api_base, api_key, data)
                response = self._dict_to_mock_response(response_dict)
            except Exception as e:
                # Map asksageclient exceptions to AskSageError
                error_msg = str(e)
                status_code = 500
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                raise AskSageError(status_code=status_code, message=error_msg)
        else:
            # Fall back to direct httpx (legacy path - DEPRECATED)
            # E37 Phase 6: This path is deprecated and will be removed in a future version
            warnings.warn(
                "Direct httpx fallback for AskSage is deprecated. "
                "Install asksageclient>=1.42 for full feature support: "
                "pip install asksageclient",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "AskSageClient not available, using deprecated async httpx fallback",
                extra={"model": model, "api_base": api_base},
            )
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = self._get_async_httpx_client(api_base=api_base, timeout=timeout)

            start_time = time.time()
            try:
                response = await client.post(
                    api_base, headers=headers, json=data, timeout=timeout
                )
                response.raise_for_status()
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="success", method="httpx_async", duration=duration
                )
            except httpx.HTTPStatusError as e:
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="error", method="httpx_async", duration=duration
                )
                error_headers = getattr(e, "headers", None)
                error_response = getattr(e, "response", None)
                if error_headers is None and error_response:
                    error_headers = getattr(error_response, "headers", None)

                raise AskSageError(
                    status_code=e.response.status_code,
                    message=await e.response.aread()
                    if hasattr(e.response, "aread")
                    else e.response.text,
                    headers=error_headers,
                )
            except Exception as e:
                duration = time.time() - start_time
                _record_request_metrics(
                    model=model, status="error", method="httpx_async", duration=duration
                )
                raise AskSageError(status_code=500, message=str(e))

        # Transform response
        config = AskSageConfig()
        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )
