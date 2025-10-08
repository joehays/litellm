"""
Common utilities for AskSage provider
"""
from typing import Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class AskSageError(BaseLLMException):
    """
    Exception raised for AskSage API errors.

    Handles error mapping from AskSage to LiteLLM error format.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or {}
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )
