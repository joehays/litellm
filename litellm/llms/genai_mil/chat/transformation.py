"""
GenAI.mil chat completion transformation

Handles request/response transformation for the GenAI.mil API.
Since GenAI.mil is OpenAI-compatible, this mostly delegates to
the OpenAIGPTConfig base class.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import (
    GENAI_MIL_API_BASE,
    GENAI_MIL_API_KEY_ENV,
    transform_model_name,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GenAIMilChatConfig(OpenAIGPTConfig):
    """
    Configuration for GenAI.mil chat completions.

    GenAI.mil is OpenAI-compatible, so this extends OpenAIGPTConfig
    and overrides provider-specific methods.
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for GenAI.mil.

        Resolution order:
        - api_base: parameter → GENAI_MIL_API_BASE env → default
        - api_key: parameter → GENAI_MIL_API_KEY env → STARK_API_KEY env

        Args:
            api_base: Optional API base URL override
            api_key: Optional API key override

        Returns:
            Tuple of (api_base, api_key)
        """
        # Resolve API base
        if api_base is None:
            api_base = get_secret_str("GENAI_MIL_API_BASE")
        if api_base is None:
            api_base = GENAI_MIL_API_BASE

        # Resolve API key (STARK token)
        if api_key is None:
            api_key = get_secret_str(GENAI_MIL_API_KEY_ENV)
        if api_key is None:
            api_key = get_secret_str("STARK_API_KEY")
        if api_key is None:
            api_key = ""  # GenAI.mil requires auth, but let validation handle it

        return api_base, api_key

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Return list of supported OpenAI parameters for GenAI.mil.

        GenAI.mil supports standard OpenAI chat completion parameters.
        """
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "n",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "logprobs",
            "top_logprobs",
        ]

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for GenAI.mil API.

        Transforms the model name (strips genai_mil/ prefix) and
        delegates to OpenAI transformation.

        Args:
            model: LiteLLM model name (e.g., genai_mil/claude-3-5-sonnet)
            messages: List of message objects
            optional_params: Additional parameters
            litellm_params: LiteLLM-specific parameters
            headers: HTTP headers

        Returns:
            Request payload dict
        """
        # Transform model name for GenAI.mil API
        genai_mil_model = transform_model_name(model)

        # Build request using OpenAI format
        data = {
            "model": genai_mil_model,
            "messages": messages,
        }

        # Add optional parameters
        for key, value in optional_params.items():
            if value is not None:
                data[key] = value

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform GenAI.mil response to LiteLLM format.

        Since GenAI.mil returns OpenAI-compatible responses,
        this parses the JSON and creates a ModelResponse.

        Args:
            model: Model name
            raw_response: HTTP response from GenAI.mil
            model_response: ModelResponse object to populate
            logging_obj: Logging object
            request_data: Original request data
            messages: Original messages
            optional_params: Additional parameters
            litellm_params: LiteLLM parameters
            encoding: Token encoding
            api_key: API key used
            json_mode: Whether JSON mode is enabled

        Returns:
            Populated ModelResponse
        """
        response_json = raw_response.json()

        # Log the response
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        # Create ModelResponse from JSON (OpenAI-compatible format)
        returned_response = ModelResponse(**response_json)

        # Add provider prefix to model name for tracking
        if returned_response.model:
            returned_response.model = f"genai_mil/{returned_response.model}"

        return returned_response

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        """
        Map OpenAI parameters to GenAI.mil parameters.

        GenAI.mil supports most OpenAI parameters directly.
        """
        mapped_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # GenAI.mil prefers max_tokens over max_completion_tokens
        if (
            "max_completion_tokens" in non_default_params
            and replace_max_completion_tokens_with_max_tokens
        ):
            mapped_params["max_tokens"] = non_default_params["max_completion_tokens"]
            mapped_params.pop("max_completion_tokens", None)

        return mapped_params
