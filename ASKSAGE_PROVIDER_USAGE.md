# AskSage Provider Usage Guide

This guide explains how to use the AskSage provider for LiteLLM to access AskSage and CAPRA endpoints.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Token Management](#token-management)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Standard AskSage (api.asksage.ai)

```python
import litellm

response = litellm.completion(
    model="asksage/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-api-key",
    api_base="https://api.asksage.ai/server/query"
)
print(response.choices[0].message.content)
```

### CAPRA (DoD Endpoints)

```python
import litellm
import os

# Configure CAPRA
os.environ["ASKSAGE_API_KEY"] = "your-capra-token"
os.environ["ASKSAGE_CA_CERT_PATH"] = "/path/to/dod-pke-ca-chain.pem"

response = litellm.completion(
    model="asksage/gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    api_base="https://api.capra.flankspeed.us.navy.mil/server/query"
)
print(response.choices[0].message.content)
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ASKSAGE_API_KEY` | Optional* | Static API token (Bearer token) |
| `ASKSAGE_TOKEN_COMMAND` | Optional* | Path to script that outputs fresh token |
| `ASKSAGE_API_BASE` | Optional | Base URL for AskSage/CAPRA endpoint |
| `ASKSAGE_CA_CERT_PATH` | CAPRA only | Path to DoD CA certificate chain |

**\*Note**: Either `ASKSAGE_API_KEY` or `ASKSAGE_TOKEN_COMMAND` must be set.

### Default Values

```python
# Default API base (CAPRA endpoint)
ASKSAGE_API_BASE = "https://api.capra.flankspeed.us.navy.mil/server/query"

# Token cache TTL
TOKEN_CACHE_TTL = 300  # 5 minutes
```

---

## Token Management

AskSage provider supports **dynamic token refresh** for CAPRA's JWT tokens that expire after 24 hours.

### Token Resolution Order

1. **Check cache** - Returns cached token if still valid (< 5 minutes old)
2. **Execute script** - Runs `ASKSAGE_TOKEN_COMMAND` if configured
3. **Static token** - Falls back to `ASKSAGE_API_KEY` environment variable
4. **Return None** - If all methods fail

### Option 1: Dynamic Token Script (Recommended for CAPRA)

Create a script that outputs a fresh token:

```bash
#!/bin/bash
# get_capra_token.sh
# Your token acquisition logic here
echo "your-fresh-jwt-token"
```

Configure LiteLLM to use the script:

```bash
export ASKSAGE_TOKEN_COMMAND="$HOME/scripts/get_capra_token.sh"
export ASKSAGE_CA_CERT_PATH="$HOME/certs/dod-pke-ca-chain.pem"
```

**Benefits**:
- Automatic token refresh
- No manual token management
- Works with 24-hour CAPRA token expiry
- Tokens cached for 5 minutes to reduce script calls

**Script Requirements**:
- Must be executable (`chmod +x script.sh`)
- Must output token to stdout
- Must complete within 10 seconds
- Should output only the token (no extra text)

### Option 2: Static Token (Simple, but requires manual refresh)

```bash
export ASKSAGE_API_KEY="your-static-bearer-token"
```

**Note**: For CAPRA, tokens expire after 24 hours. You'll need to manually update this environment variable.

### Token Caching

Tokens are cached for **5 minutes** to reduce:
- Script execution overhead
- API token validation calls
- System load

Cache is cleared automatically after TTL expires or can be manually cleared by restarting your application.

---

## Usage Examples

### Basic Completion

```python
import litellm

response = litellm.completion(
    model="asksage/gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### With System Prompt

```python
response = litellm.completion(
    model="asksage/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful physics teacher."},
        {"role": "user", "content": "What is relativity?"}
    ]
)
```

### With RAG Context (Dataset)

```python
response = litellm.completion(
    model="asksage/claude-3-opus",
    messages=[{"role": "user", "content": "What is our return policy?"}],
    dataset=["company_policies_2024"]  # RAG context
)
```

### With Persona

```python
response = litellm.completion(
    model="asksage/gpt-4",
    messages=[{"role": "user", "content": "Review this code"}],
    persona="senior_engineer"  # AI personality
)
```

### With Temperature Control

```python
response = litellm.completion(
    model="asksage/gpt-4",
    messages=[{"role": "user", "content": "Generate creative story ideas"}],
    temperature=0.9  # Higher = more creative
)
```

### Async Completion

```python
import asyncio
import litellm

async def get_response():
    response = await litellm.acompletion(
        model="asksage/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello async!"}]
    )
    return response

response = asyncio.run(get_response())
```

### List Available Models

```python
import litellm

# List all AskSage models
models = [m for m in litellm.model_list if m.startswith("asksage/")]
print(f"Available AskSage models: {len(models)}")
```

---

## Troubleshooting

### Error: "No authentication token provided"

**Cause**: Neither `ASKSAGE_API_KEY` nor `ASKSAGE_TOKEN_COMMAND` is set.

**Solution**:
```bash
export ASKSAGE_TOKEN_COMMAND="$HOME/scripts/get_capra_token.sh"
# OR
export ASKSAGE_API_KEY="your-token-here"
```

### Error: "Token is invalid" or "Token expired"

**Cause**: CAPRA tokens expire after 24 hours.

**Solutions**:
1. **Use dynamic token script** (recommended):
   ```bash
   export ASKSAGE_TOKEN_COMMAND="$HOME/scripts/get_capra_token.sh"
   ```

2. **Manually refresh static token**:
   ```bash
   export ASKSAGE_API_KEY="your-new-token-here"
   ```

### Error: "SSL verification failed" or "Certificate error"

**Cause**: CAPRA requires DoD CA certificate chain.

**Solution**:
```bash
export ASKSAGE_CA_CERT_PATH="/path/to/dod-pke-ca-chain.pem"
```

**Download DoD Certificates**:
- Visit: https://public.cyber.mil/pki-pke/
- Download: "DoD PKE CA Certificates (PKCS#7)"
- Extract: dod-pke-ca-chain.pem

### Error: "ASKSAGE_TOKEN_COMMAND timed out"

**Cause**: Token script took longer than 10 seconds.

**Solutions**:
1. Optimize your token script
2. Use static token as fallback:
   ```bash
   export ASKSAGE_TOKEN_COMMAND="$HOME/scripts/get_capra_token.sh"
   export ASKSAGE_API_KEY="fallback-token"  # Used if script fails
   ```

### Error: "ASKSAGE_TOKEN_COMMAND failed"

**Cause**: Token script returned non-zero exit code or error.

**Debug**:
```bash
# Test script manually
bash -x $ASKSAGE_TOKEN_COMMAND

# Check script permissions
chmod +x $ASKSAGE_TOKEN_COMMAND
```

### Token Script Not Being Called

**Cause**: Token might be cached.

**Solution**: Wait 5 minutes for cache to expire or restart application.

**Verify cache**:
```python
from litellm.llms.asksage.common_utils import _token_cache
_token_cache.clear()  # Force refresh
```

---

## Advanced Configuration

### Custom Cache TTL

Modify cache TTL by setting environment variable before importing litellm:

```python
import os
os.environ["ASKSAGE_TOKEN_CACHE_TTL"] = "600"  # 10 minutes

import litellm
# Now cache will use 10-minute TTL
```

### CAPRA Full Example

Complete setup for CAPRA with all features:

```bash
#!/bin/bash
# setup_capra.sh

# Token refresh script
export ASKSAGE_TOKEN_COMMAND="$HOME/scripts/get_capra_token.sh"

# Static fallback (optional)
export ASKSAGE_API_KEY="fallback-static-token"

# CAPRA endpoint
export ASKSAGE_API_BASE="https://api.capra.flankspeed.us.navy.mil/server/query"

# DoD TLS certificate
export ASKSAGE_CA_CERT_PATH="$HOME/certs/dod-pke-ca-chain.pem"

# Run your application
python your_app.py
```

```python
# your_app.py
import litellm

response = litellm.completion(
    model="asksage/gpt-4o-mini",
    messages=[{"role": "user", "content": "Test CAPRA"}],
    dataset=["none"],
    temperature=0.7
)
print(response.choices[0].message.content)
```

---

## Performance Notes

- **Token script execution**: ~12ms (fast enough for real-time use)
- **Token caching**: Tokens cached for 5 minutes to reduce overhead
- **Async support**: Use `acompletion()` for non-blocking operations
- **Connection pooling**: httpx handles connection reuse automatically

---

## Security Best Practices

1. **Never commit tokens**: Add token scripts to `.gitignore`
2. **File permissions**: Restrict token script access (`chmod 700`)
3. **Environment variables**: Use environment variables, not hardcoded tokens
4. **Token rotation**: Use dynamic token scripts for automatic rotation
5. **Certificate validation**: Always use `ASKSAGE_CA_CERT_PATH` for CAPRA

---

## Support

For issues or questions:
- LiteLLM GitHub: https://github.com/BerriAI/litellm/issues
- AskSage Community: https://github.com/Ask-Sage/AskSage-Open-Source-Community
- MARS Project: Contact your team lead

---

## Version

- **AskSage Provider**: v1.0
- **LiteLLM**: Compatible with v1.x+
- **Last Updated**: 2025-10-08
