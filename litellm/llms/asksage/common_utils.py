"""
Common utilities for AskSage provider
"""
import os
import subprocess
import time
from typing import Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class AskSageError(BaseLLMException):
    """
    Exception raised for AskSage API errors.

    Handles error mapping from AskSage to LiteLLM error format.
    """

    def __init__(self, status_code: int, message: str, headers: Optional[dict] = None):
        self.status_code = status_code
        self.message = message
        self.headers = headers or {}
        super().__init__(status_code=status_code, message=message, headers=self.headers)


class TokenCache:
    """
    Simple token cache with TTL to avoid calling token script on every request.

    Default TTL is 5 minutes, which is safe since CAPRA tokens last 24 hours.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.token: Optional[str] = None
        self.timestamp: float = 0.0
        self.ttl_seconds = ttl_seconds

    def get(self) -> Optional[str]:
        """Get cached token if still valid"""
        if self.token and (time.time() - self.timestamp) < self.ttl_seconds:
            return self.token
        return None

    def set(self, token: str) -> None:
        """Cache a token"""
        self.token = token
        self.timestamp = time.time()

    def clear(self) -> None:
        """Clear cached token"""
        self.token = None
        self.timestamp = 0.0


# Global token cache instance
_token_cache = TokenCache()


def get_asksage_token() -> Optional[str]:
    """
    Get AskSage/CAPRA API token.

    Token resolution order:
    1. Check cache (if token was recently fetched)
    2. Execute ASKSAGE_TOKEN_COMMAND script if configured
    3. Use ASKSAGE_API_KEY environment variable
    4. Return None

    Environment Variables:
        ASKSAGE_TOKEN_COMMAND: Path to script that outputs token (e.g., ~/dev/joe-docs/dev-ops/get_capra_access_token.sh)
        ASKSAGE_API_KEY: Static token (fallback if no script configured)
        ASKSAGE_TOKEN_CACHE_TTL: Cache TTL in seconds (default: 300)

    Returns:
        Token string or None
    """
    # Check cache first
    cached_token = _token_cache.get()
    if cached_token:
        return cached_token

    # Try to execute token command script
    token_command = os.environ.get("ASKSAGE_TOKEN_COMMAND")
    if token_command:
        try:
            # Expand ~ and environment variables in path
            token_command = os.path.expanduser(token_command)
            token_command = os.path.expandvars(token_command)

            # Execute script and capture output
            result = subprocess.run(
                [token_command],
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout for safety
                check=True,
            )

            # Get token from stdout, strip whitespace
            token = result.stdout.strip()

            if token:
                # Cache the token
                _token_cache.set(token)
                return token

        except subprocess.TimeoutExpired:
            # Log warning but fall through to static token
            import litellm

            litellm.verbose_logger.warning(
                f"ASKSAGE_TOKEN_COMMAND timed out: {token_command}"
            )
        except subprocess.CalledProcessError as e:
            # Log warning but fall through to static token
            import litellm

            litellm.verbose_logger.warning(
                f"ASKSAGE_TOKEN_COMMAND failed: {token_command} - {e}"
            )
        except Exception as e:
            # Log warning but fall through to static token
            import litellm

            litellm.verbose_logger.warning(
                f"ASKSAGE_TOKEN_COMMAND error: {token_command} - {e}"
            )

    # Fall back to static token
    static_token = os.environ.get("ASKSAGE_API_KEY")
    if static_token:
        # Cache static token too (with shorter TTL since it doesn't refresh)
        _token_cache.set(static_token)
        return static_token

    return None
