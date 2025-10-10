"""
Transformation logic for AskSage chat completions

Handles request/response transformation between LiteLLM and AskSage API formats.
"""
import json
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.secret_managers.main import get_secret
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import Choices, Message

from ..common_utils import AskSageError


class AskSageConfig(BaseConfig):
    """
    Configuration for AskSage provider

    Reference: https://api.asksage.ai/documentation/
    CAPRA: https://api.capra.flankspeed.us.navy.mil/
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def __init__(
        self, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "asksage"

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Return list of OpenAI parameters supported by AskSage
        """
        return [
            "temperature",
            "max_tokens",
            "stream",  # TODO: Verify if AskSage supports streaming
        ]

    def map_openai_params(
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool = False,
    ) -> Dict:
        """
        Map OpenAI parameters to AskSage format

        Args:
            non_default_params: Non-default parameters from request
            optional_params: Optional parameters dict
            model: Model name
            drop_params: Whether to drop unsupported params

        Returns:
            Mapped parameters dict
        """
        # AskSage uses same parameter names as OpenAI for these
        supported = self.get_supported_openai_params(model)

        for param, value in non_default_params.items():
            if param in supported:
                optional_params[param] = value

        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> Any:
        """
        Return appropriate error class for given status code

        Args:
            error_message: Error message from API
            status_code: HTTP status code
            headers: Response headers

        Returns:
            AskSageError instance
        """
        return AskSageError(
            status_code=status_code,
            message=error_message,
            headers=dict(headers) if isinstance(headers, httpx.Headers) else headers,
        )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Validate environment and set up authentication headers

        Args:
            headers: Existing headers dict
            model: Model name
            messages: Message list
            optional_params: Optional parameters
            api_key: Bearer token for AskSage/CAPRA
            api_base: Base URL for AskSage API

        Returns:
            Updated headers dict with x-access-tokens
        """
        # Set x-access-tokens header (CAPRA-specific, no "Bearer" prefix)
        if api_key:
            headers["x-access-tokens"] = api_key
            print(
                f"[DEBUG] AskSage: Set x-access-tokens header with token: {api_key[:50]}..."
            )
        else:
            print("[DEBUG] AskSage: No api_key provided!")

        headers["Content-Type"] = "application/json"
        print(f"[DEBUG] AskSage: Final headers: {list(headers.keys())}")

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        headers: Dict,
    ) -> Dict:
        """
        Transform LiteLLM request to AskSage API format

        LiteLLM format:
        - messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        - model: "asksage/google-claude-4-opus"
        - optional_params: {"dataset": ["ds_123"], "persona": "researcher", ...}

        AskSage format:
        - message: "user message text"
        - model: "google-claude-4-opus" (optional)
        - dataset_ids: ["ds_123"] (optional)
        - persona: "researcher" (optional)
        - system_prompt: "system message text" (optional)
        - temperature: 0.7 (optional)
        """
        # Extract message content from messages array
        # Combine system messages into system_prompt, use last user message as message
        system_messages = []
        user_message = ""

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_messages.append(content)
            elif role == "user":
                user_message = content  # Use last user message
            elif role == "assistant":
                # AskSage doesn't support conversation history in single endpoint
                # We'll append assistant message to context if needed
                pass

        # Build AskSage request payload
        data: Dict[str, Any] = {
            "message": user_message,
            "dataset": optional_params.get(
                "dataset", ["none"]
            ),  # AskSage requires dataset array
        }

        # Add model if specified (optional in AskSage)
        if model:
            data["model"] = model

        # Add system prompt if any system messages exist
        if system_messages:
            data["system_prompt"] = "\n".join(system_messages)

        # Add custom system_prompt from optional_params (overrides extracted system messages)
        if "system_prompt" in optional_params:
            data["system_prompt"] = optional_params["system_prompt"]

        # Add persona if specified
        if "persona" in optional_params:
            data["persona"] = optional_params["persona"]

        # Add temperature
        if "temperature" in optional_params:
            data["temperature"] = optional_params["temperature"]

        # Add max_tokens if supported (TODO: verify AskSage parameter name)
        if "max_tokens" in optional_params:
            data["max_tokens"] = optional_params["max_tokens"]

        # Add limit_references if specified
        if "limit_references" in optional_params:
            data["limit_references"] = optional_params["limit_references"]

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform AskSage response to LiteLLM ModelResponse format

        AskSage format:
        {
            "response": "AI-generated response...",
            "model_used": "google-claude-4-opus",
            "tokens_used": {
                "prompt": 15,
                "completion": 150,
                "total": 165
            },
            "citations": [...]
        }

        LiteLLM format: ModelResponse with choices and usage
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise AskSageError(
                status_code=raw_response.status_code,
                message=f"Failed to parse AskSage response: {str(e)}",
            )

        # Extract response text
        response_text = response_json.get("response", "")

        # Extract model used
        model_used = response_json.get("model_used", model)

        # Extract token usage
        tokens_used = response_json.get("tokens_used", {})
        usage = Usage(
            prompt_tokens=tokens_used.get("prompt", 0),
            completion_tokens=tokens_used.get("completion", 0),
            total_tokens=tokens_used.get("total", 0),
        )

        # Build message
        message = Message(content=response_text, role="assistant")

        # Store citations in metadata if present
        citations = response_json.get("citations")
        if citations:
            # Store in model_response metadata or as custom field
            if not hasattr(model_response, "_hidden_params"):
                model_response._hidden_params = {}
            model_response._hidden_params["citations"] = citations

        # Build choice
        choice = Choices(
            finish_reason=map_finish_reason(
                "stop"
            ),  # AskSage doesn't provide finish_reason
            index=0,
            message=message,
        )

        # Update model_response
        model_response.choices = [choice]
        model_response.model = model_used
        model_response.usage = usage

        return model_response
