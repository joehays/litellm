"""
GenAI.mil chat completion module
"""

from .handler import GenAIMilChatCompletion
from .transformation import GenAIMilChatConfig

__all__ = [
    "GenAIMilChatCompletion",
    "GenAIMilChatConfig",
]
