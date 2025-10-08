"""
Handler for AskSage chat completions

Implements the chat completion API for AskSage/CAPRA.
"""
import json
import os
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


class AskSageChatCompletion(BaseLLM):
    """
    Handler for AskSage chat completion API

    Supports both standard AskSage (https://api.asksage.ai) and
    CAPRA (https://api.capra.flankspeed.us.navy.mil) endpoints.
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_httpx_client(
        self,
        api_base: str,
        timeout: Union[float, httpx.Timeout],
    ) -> httpx.Client:
        """
        Create httpx client with optional TLS certificate for CAPRA

        Checks for ASKSAGE_CA_CERT_PATH environment variable.
        If present, uses it for TLS verification (required for CAPRA/DoD endpoints).
        """
        ca_cert_path = os.environ.get("ASKSAGE_CA_CERT_PATH")

        if ca_cert_path and os.path.exists(ca_cert_path):
            # Use DoD CA certificate chain for CAPRA
            return httpx.Client(
                timeout=timeout,
                verify=ca_cert_path,
            )
        else:
            # Standard HTTPS (use system CA bundle)
            return _get_httpx_client(params={"timeout": timeout})

    def _get_async_httpx_client(
        self,
        api_base: str,
        timeout: Union[float, httpx.Timeout],
    ) -> httpx.AsyncClient:
        """
        Create async httpx client with optional TLS certificate for CAPRA
        """
        ca_cert_path = os.environ.get("ASKSAGE_CA_CERT_PATH")

        if ca_cert_path and os.path.exists(ca_cert_path):
            # Use DoD CA certificate chain for CAPRA
            return httpx.AsyncClient(
                timeout=timeout,
                verify=ca_cert_path,
            )
        else:
            # Standard HTTPS (use system CA bundle)
            return get_async_httpx_client(
                llm_provider=litellm.LlmProviders.ASKSAGE,
                params={"timeout": timeout},
            )

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

        print(f"[DEBUG] AskSage handler.completion() called!")
        print(f"[DEBUG]   api_base: {api_base}")
        print(f"[DEBUG]   api_key: {api_key[:50] if api_key else 'NONE'}...")
        print(f"[DEBUG]   model: {model}")

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

        # Synchronous completion
        if client is None or not isinstance(client, HTTPHandler):
            client = self._get_httpx_client(api_base=api_base, timeout=timeout)

        try:
            response = client.post(
                api_base,
                headers=headers,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
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
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = self._get_async_httpx_client(api_base=api_base, timeout=timeout)

        try:
            response = await client.post(
                api_base,
                headers=headers,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_headers = getattr(e, "headers", None)
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)

            raise AskSageError(
                status_code=e.response.status_code,
                message=await e.response.aread() if hasattr(e.response, "aread") else e.response.text,
                headers=error_headers,
            )
        except Exception as e:
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
