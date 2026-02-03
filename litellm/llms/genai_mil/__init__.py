"""
GenAI.mil (US Government AI Gateway) LiteLLM Provider

GenAI.mil is an OpenAI-compatible API gateway for US Government AI services.
It provides access to multiple LLM providers (Claude, Gemini, GPT) through a
unified API at https://api.genai.mil/v1.

Authentication:
    Uses STARK tokens via GENAI_MIL_API_KEY environment variable.

Model Naming:
    Models are prefixed with 'genai/' in LiteLLM:
    - genai/claude-3-5-sonnet → claude-3-5-sonnet-20241022 on GenAI.mil
    - genai/gemini-2.5-pro → gemini-2.5-pro on GenAI.mil
    - genai/gpt-4o → gpt-4o on GenAI.mil

Usage:
    import litellm

    response = litellm.completion(
        model="genai/claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key=os.environ.get("GENAI_MIL_API_KEY"),
    )

Configuration (via litellm config.yaml):
    model_list:
      - model_name: genai/claude-3-5-sonnet
        litellm_params:
          model: genai_mil/claude-3-5-sonnet-20241022
          api_key: os.environ/GENAI_MIL_API_KEY

Reference:
    - GenAI.mil: https://api.genai.mil
    - Implementation: MARS Issue #40
"""

from .common_utils import GenAIMilError
from .chat.handler import GenAIMilChatCompletion
from .chat.transformation import GenAIMilChatConfig

__all__ = [
    "GenAIMilError",
    "GenAIMilChatCompletion",
    "GenAIMilChatConfig",
]
