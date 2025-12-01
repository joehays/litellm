# Safe Testing Guide for LiteLLM Routing

This guide describes how to safely validate LiteLLM routing behavior without consuming paid API tokens.

## Prerequisites

- Local Ollama running (port 11434 or 11435)
- LiteLLM proxy running with Ollama-only config
- Python with `requests` and `pytest` installed

## Test Approaches

### Approach 1: Ollama Log Counting

Monitor Ollama container logs to count actual backend calls:

```python
def get_ollama_request_count(since_seconds: int = 30) -> int:
    """Count requests in Ollama container logs."""
    result = subprocess.run(
        ["docker", "logs", "mars-ollama", "--since", f"{since_seconds}s"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "DOCKER_HOST": ""}
    )
    logs = result.stdout + result.stderr
    # Ollama logs "POST /api/chat" or "POST /api/generate" for each request
    return logs.count("POST /api/")
```

**Advantages**:
- Uses real infrastructure
- No mock setup required
- Validates end-to-end behavior

**Limitations**:
- Requires running Ollama container
- Log format might change
- Timing-sensitive (need to wait for logs)

### Approach 2: Mock HTTP Server

Run a mock server that counts all incoming requests:

```python
class MockLLMHandler(BaseHTTPRequestHandler):
    """HTTP handler that simulates LLM API and logs requests."""

    def do_POST(self):
        # Log the request to global state
        _mock_state.add_request(RequestLog(...))

        # Return mock LLM response
        response = {
            "id": "mock-response-id",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Mock response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        self.send_response(200)
        self.wfile.write(json.dumps(response).encode())
```

**Advantages**:
- Complete control over response
- Fast (no actual LLM inference)
- Definitive request counting

**Limitations**:
- Requires custom LiteLLM config pointing to mock
- May not catch all edge cases

## Test Configuration

### Ollama-Only Config (Safe)

```yaml
model_list:
  - model_name: test-model-1
    litellm_params:
      model: ollama/qwen2.5-coder:7b
      api_base: http://localhost:11434

  - model_name: test-model-2
    litellm_params:
      model: ollama/qwen2.5-coder:7b
      api_base: http://localhost:11434

  - model_name: test-model-3
    litellm_params:
      model: ollama/qwen2.5-coder:7b
      api_base: http://localhost:11434

litellm_settings:
  set_verbose: true
  cache: false
  num_retries: 0

router_settings:
  routing_strategy: "simple-shuffle"
```

### Mock Server Config

```yaml
model_list:
  - model_name: test-model-0
    litellm_params:
      model: openai/gpt-test-0
      api_base: http://localhost:18199
      api_key: test-key

  # Repeat for multiple "models" all pointing to same mock

litellm_settings:
  drop_params: true
  cache: false
  num_retries: 0
```

## Key Test Cases

### Test 1: Single Model Request = Single Backend Call

```python
def test_single_request_single_backend():
    """1 LiteLLM request should result in exactly 1 backend call."""
    baseline = get_request_count()

    make_request(model="test-model-1")

    new_calls = get_request_count() - baseline
    assert new_calls == 1, f"Expected 1 call, got {new_calls}"
```

### Test 2: Multiple Models Configured, Only One Called

```python
def test_multiple_models_one_called():
    """With 5 models configured, only the requested model should be called."""
    # Configure 5 models pointing to mock server
    # Make 1 request to specific model
    # Assert mock received exactly 1 request
```

### Test 3: Rate Limiting Enforced

```python
def test_rate_limiting():
    """max_parallel_requests should limit concurrent calls."""
    # Configure max_parallel_requests: 2
    # Fire 10 requests in parallel
    # Verify rate limiting queued/limited appropriately
```

## Running Tests

```bash
# Run Ollama-based tests (requires running Ollama)
pytest tests/regression/test_litellm_single_model_routing.py -v -m integration

# Run mock server tests (standalone)
pytest tests/regression/test_litellm_call_counting.py -v

# Run all routing tests
pytest tests/regression/test_litellm*.py -v --tb=short
```

## Approach 3: Token Counting Validator (NEW - December 2024)

A dedicated validation utility that measures token consumption before/after requests:

```python
# Safe Ollama testing (default)
python modules/services/litellm/tests/validation/token_counting_validator.py --backend ollama

# Multiple requests
python modules/services/litellm/tests/validation/token_counting_validator.py --backend ollama --count 3

# AskSage testing (uses real tokens - requires confirmation)
python modules/services/litellm/tests/validation/token_counting_validator.py --backend asksage
```

**For Ollama**: Uses `response.usage` field to track tokens
**For AskSage**: Uses `GET /count-monthly-tokens` API endpoint

### AskSage Token Tracking API Endpoints

Discovered in AskSage documentation:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/count-monthly-tokens` | GET | Returns count of tokens used this month |
| `/count-monthly-tokens` | POST | Returns token count for specific app |
| `/count-monthly-teach-tokens` | GET | Returns training tokens used this month |
| `/get-user-logs` | POST | Get your last prompts (User API) |

### Pre/Post Token Counting Strategy for AskSage

1. **Before test**: Call `GET /count-monthly-tokens` to record baseline
2. **Execute single request**: Make one LiteLLM request to single AskSage model
3. **After test**: Call `GET /count-monthly-tokens` again
4. **Validate**: Compare delta - should match expected single-request consumption

If delta is unexpectedly large (>5x expected), the 38x routing bug may still be present.

## Running Tests

```bash
# Run Ollama-based tests (requires running Ollama)
pytest tests/regression/test_litellm_single_model_routing.py -v -m integration

# Run mock server tests (standalone)
pytest tests/regression/test_litellm_call_counting.py -v

# Run token counting validation tests
pytest modules/services/litellm/tests/validation/token_counting_validator.py -v

# Run all routing tests
pytest tests/regression/test_litellm*.py -v --tb=short
```

## Interpreting Results

### Pass Criteria
- 1 LiteLLM request = 1 backend call
- No spurious calls to unconfigured models
- Rate limiting properly enforced
- Token consumption within expected range (10-500 for simple requests)

### Failure Indicators
- Mock server receives N calls for 1 request (where N > 1)
- Logs show multiple model names for single request
- Rate limiter bypassed
- Token consumption >5x expected (indicates routing bug)
