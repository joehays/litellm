"""
GenAI.mil chat completion handler

Handles chat completion requests to the GenAI.mil API.
Since GenAI.mil is OpenAI-compatible, this extends the
OpenAILikeChatHandler with GenAI.mil-specific configuration.
"""

import json
import logging
from typing import Any, Callable, Optional, Union

import httpx

import litellm
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.utils import CustomStreamingDecoder, ModelResponse
from litellm.utils import CustomStreamWrapper

from ..common_utils import GenAIMilBase, GenAIMilError, transform_model_name
from .transformation import GenAIMilChatConfig

# Set up logging
logger = logging.getLogger("litellm.genai_mil")


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    streaming_decoder: Optional[CustomStreamingDecoder] = None,
    fake_stream: bool = False,
):
    """
    Make async streaming request to GenAI.mil.

    Args:
        client: Async HTTP client
        api_base: API endpoint URL
        headers: Request headers
        data: JSON request body
        model: Model name
        messages: Original messages
        logging_obj: Logging object
        streaming_decoder: Optional custom streaming decoder
        fake_stream: Whether to simulate streaming from non-streaming response

    Returns:
        Streaming response iterator
    """
    if client is None:
        client = litellm.module_level_aclient

    response = await client.post(
        api_base, headers=headers, data=data, stream=not fake_stream
    )

    if streaming_decoder is not None:
        completion_stream: Any = streaming_decoder.aiter_bytes(
            response.aiter_bytes(chunk_size=1024)
        )
    elif fake_stream:
        model_response = ModelResponse(**response.json())
        completion_stream = MockResponseIterator(model_response=model_response)
    else:
        completion_stream = ModelResponseIterator(
            streaming_response=response.aiter_lines(), sync_stream=False
        )

    # Log the call
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    streaming_decoder: Optional[CustomStreamingDecoder] = None,
    fake_stream: bool = False,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
):
    """
    Make sync streaming request to GenAI.mil.

    Args:
        client: Sync HTTP client
        api_base: API endpoint URL
        headers: Request headers
        data: JSON request body
        model: Model name
        messages: Original messages
        logging_obj: Logging object
        streaming_decoder: Optional custom streaming decoder
        fake_stream: Whether to simulate streaming from non-streaming response
        timeout: Request timeout

    Returns:
        Streaming response iterator
    """
    if client is None:
        client = litellm.module_level_client

    response = client.post(
        api_base, headers=headers, data=data, stream=not fake_stream, timeout=timeout
    )

    if response.status_code != 200:
        raise GenAIMilError(status_code=response.status_code, message=response.read())

    if streaming_decoder is not None:
        completion_stream = streaming_decoder.iter_bytes(
            response.iter_bytes(chunk_size=1024)
        )
    elif fake_stream:
        model_response = ModelResponse(**response.json())
        completion_stream = MockResponseIterator(model_response=model_response)
    else:
        completion_stream = ModelResponseIterator(
            streaming_response=response.iter_lines(), sync_stream=True
        )

    # Log the call
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class GenAIMilChatCompletion(GenAIMilBase):
    """
    Handler for GenAI.mil chat completions.

    GenAI.mil is OpenAI-compatible, so this handler uses standard
    OpenAI request/response formats with GenAI.mil-specific authentication.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        custom_llm_provider: str,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ) -> CustomStreamWrapper:
        """Handle async streaming completion."""
        data["stream"] = True

        completion_stream = await make_call(
            client=client,
            api_base=api_base,
            headers=headers,
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            streaming_decoder=streaming_decoder,
        )

        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

    async def acompletion_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        custom_llm_provider: str,
        print_verbose: Callable,
        client: Optional[AsyncHTTPHandler],
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
        base_model: Optional[str],
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Handle async non-streaming completion."""
        if timeout is None:
            timeout = httpx.Timeout(timeout=600.0, connect=5.0)

        if client is None:
            client = litellm.module_level_aclient

        try:
            response = await client.post(
                api_base, headers=headers, data=json.dumps(data), timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GenAIMilError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise GenAIMilError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise GenAIMilError(status_code=500, message=str(e))

        # Transform response
        config = GenAIMilChatConfig()
        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

    def completion(
        self,
        *,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params: dict = {},
        logger_fn=None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        """
        Main completion handler for GenAI.mil.

        Args:
            model: Model name (e.g., genai_mil/claude-3-5-sonnet)
            messages: List of message dicts
            api_base: GenAI.mil API base URL
            api_key: STARK token for authentication
            optional_params: Additional parameters
            acompletion: If True, use async completion

        Returns:
            ModelResponse or CustomStreamWrapper for streaming
        """
        # Handle custom endpoint setting
        custom_endpoint = custom_endpoint or optional_params.pop(
            "custom_endpoint", None
        )
        base_model: Optional[str] = optional_params.pop("base_model", None)

        # Validate environment and set up headers
        api_base, headers = self._validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="chat_completions",
            custom_endpoint=custom_endpoint,
            headers=headers,
        )

        # Extract stream and other params
        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})
        json_mode = optional_params.pop("json_mode", None)
        optional_params.pop("max_retries", None)

        if not fake_stream:
            optional_params["stream"] = stream

        # Transform model name for GenAI.mil API
        genai_model = transform_model_name(model)

        # Build request data
        data = {
            "model": genai_model,
            "messages": messages,
            **optional_params,
            **extra_body,
        }

        # Log the request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        logger.debug(
            "GenAI.mil completion request",
            extra={
                "model": genai_model,
                "api_base": api_base,
                "stream": stream,
                "acompletion": acompletion,
            },
        )

        # Handle async completion
        if acompletion is True:
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = None

            if stream is True:
                data["stream"] = stream
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=client,
                    custom_llm_provider=custom_llm_provider,
                    streaming_decoder=streaming_decoder,
                    fake_stream=fake_stream,
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    custom_llm_provider=custom_llm_provider,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    base_model=base_model,
                    client=client,
                    json_mode=json_mode,
                )

        # Handle sync completion
        if stream is True:
            completion_stream = make_sync_call(
                client=(
                    client
                    if client is not None and isinstance(client, HTTPHandler)
                    else None
                ),
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                streaming_decoder=streaming_decoder,
                fake_stream=fake_stream,
                timeout=timeout,
            )
            return CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )
        else:
            # Non-streaming sync completion
            if client is None or not isinstance(client, HTTPHandler):
                client = HTTPHandler(timeout=timeout)

            try:
                response = client.post(
                    url=api_base, headers=headers, data=json.dumps(data)
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise GenAIMilError(
                    status_code=e.response.status_code,
                    message=e.response.text,
                )
            except httpx.TimeoutException:
                raise GenAIMilError(status_code=408, message="Timeout error occurred.")
            except Exception as e:
                raise GenAIMilError(status_code=500, message=str(e))

        # Transform response
        config = GenAIMilChatConfig()
        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
