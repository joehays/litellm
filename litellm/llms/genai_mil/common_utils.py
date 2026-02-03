"""
GenAI.mil common utilities

Provides error handling and configuration utilities for the GenAI.mil provider.
"""

import os
from typing import Literal, Optional, Tuple

import httpx

# Default GenAI.mil API base URL
GENAI_MIL_API_BASE = "https://api.genai.mil/v1"

# Environment variable names
GENAI_MIL_API_KEY_ENV = "GENAI_MIL_API_KEY"
GENAI_MIL_API_BASE_ENV = "GENAI_MIL_API_BASE"


class GenAIMilError(Exception):
    """
    Exception class for GenAI.mil API errors.

    Attributes:
        status_code: HTTP status code from the API
        message: Error message
        headers: Optional response headers
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        self.request = httpx.Request(method="POST", url=GENAI_MIL_API_BASE)
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)


def get_genai_mil_api_key() -> Optional[str]:
    """
    Get GenAI.mil API key from environment.

    Resolution order:
    1. GENAI_MIL_API_KEY environment variable
    2. STARK_API_KEY environment variable (legacy)

    Returns:
        API key string or None if not configured
    """
    return os.environ.get(GENAI_MIL_API_KEY_ENV) or os.environ.get("STARK_API_KEY")


def get_genai_mil_api_base() -> str:
    """
    Get GenAI.mil API base URL from environment.

    Returns:
        API base URL (default: https://api.genai.mil/v1)
    """
    return os.environ.get(GENAI_MIL_API_BASE_ENV, GENAI_MIL_API_BASE)


class GenAIMilBase:
    """Base class for GenAI.mil handlers."""

    def __init__(self, **kwargs):
        pass

    def _validate_environment(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        headers: Optional[dict],
        custom_endpoint: Optional[bool],
    ) -> Tuple[str, dict]:
        """
        Validate and configure the request environment.

        Args:
            api_key: GenAI.mil API key (STARK token)
            api_base: API base URL (default: https://api.genai.mil/v1)
            endpoint_type: Type of endpoint (chat_completions or embeddings)
            headers: Optional custom headers
            custom_endpoint: Whether to use a custom endpoint (skip path appending)

        Returns:
            Tuple of (api_base, headers) ready for the request
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = get_genai_mil_api_key()

        if api_key is None and headers is None:
            raise GenAIMilError(
                status_code=401,
                message="Missing API Key - Set GENAI_MIL_API_KEY environment variable "
                "or provide api_key parameter",
            )

        # Get API base from parameter or environment
        if api_base is None:
            api_base = get_genai_mil_api_base()

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")

        # Set up headers
        if headers is None:
            headers = {"Content-Type": "application/json"}

        # Add Authorization header with Bearer token
        if api_key is not None and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"

        # Append endpoint path unless custom_endpoint is True
        if not custom_endpoint:
            if endpoint_type == "chat_completions":
                api_base = f"{api_base}/chat/completions"
            elif endpoint_type == "embeddings":
                api_base = f"{api_base}/embeddings"

        return api_base, headers


def transform_model_name(model: str) -> str:
    """
    Transform LiteLLM model name to GenAI.mil model name.

    Strips the 'genai_mil/' prefix if present.

    Examples:
        genai_mil/claude-3-5-sonnet-20241022 → claude-3-5-sonnet-20241022
        genai_mil/gemini-2.5-pro → gemini-2.5-pro
        gpt-4o → gpt-4o  (no change)

    Args:
        model: LiteLLM model identifier

    Returns:
        Model name for GenAI.mil API
    """
    if model.startswith("genai_mil/"):
        return model[len("genai_mil/"):]
    return model
