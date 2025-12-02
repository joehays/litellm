# Model Broadcast Bug Investigation

**Date**: 2024-12-01
**Severity**: CRITICAL
**Impact**: Potential 38x token consumption per request when multiple models configured

## Executive Summary

A configuration scenario was discovered where LiteLLM proxy routes requests to ALL configured models in parallel instead of the single requested model. This causes catastrophic token consumption when using paid API providers.

## Incident Details

### What Happened
- 6 HTTP POST requests to LiteLLM proxy
- Expected: 6 backend API calls (1 per request)
- Actual: 3,842 backend API calls (38 models x ~100 calls each)
- Token consumption: 4.9M tokens in ~60-90 minutes

### Evidence from Container Logs
```
183 [DEBUG]   model: google-claude-4-opus
102 [DEBUG]   model: google-claude-45-sonnet
 96 [DEBUG]   model: gpt-gov
 96 [DEBUG]   model: gpt4
 95 [DEBUG]   model: gpt-4o-mini
...
```

All 38 configured models called 90-100 times each for only 6 HTTP requests.

## Investigation Findings

### Router Code Analysis

The core routing logic in `router.py` appears correct:

1. **`simple_shuffle()` strategy**: Returns SINGLE deployment
   - File: `litellm/router_strategy/simple_shuffle.py`
   - Behavior: Filters healthy deployments, shuffles, returns `deployments[0]`

2. **`async_get_available_deployment()`**: Correctly picks one model
   - Uses routing strategy to select single deployment
   - No iteration over all models in main routing path

3. **No batch_completion patterns found** in logs during incident

4. **Comma-separated model handling** not triggered (single model name used)

### Ruled Out Causes

| Potential Cause | Status | Reason |
|-----------------|--------|--------|
| `simple-shuffle` routing strategy | RULED OUT | Correctly returns single model |
| `batch_completion` mode | RULED OUT | No matching log entries |
| Comma-separated models in request | RULED OUT | Test used single model name |
| Background health checks | RULED OUT | Not enabled in config |
| Router fallback cascading | RULED OUT | Only one model should be tried |

### Suspected Causes (Needs Investigation)

1. **Provider initialization/validation layer** - May iterate models during setup
2. **Async callback iteration** - Callbacks might trigger model calls
3. **Model alias resolution** - Complex alias chains might cause iteration
4. **Pre-call model verification** - May check all models before routing

## Configuration That Caused Issue

```yaml
model_list:
  # 38 models configured - ALL were called per request
  - model_name: google-claude-45-sonnet
    litellm_params:
      model: asksage/google-claude-45-sonnet
      api_base: ...
      api_key: ...
  # ... 37 more models ...

litellm_settings:
  set_verbose: true
  cache: false
```

## Mitigation Applied

### Immediate Fix
```yaml
# Disabled all external API models
# Only local Ollama models remain enabled
model_list:
  - model_name: qwen2.5-coder:7b
    litellm_params:
      model: ollama/qwen2.5-coder:7b
      api_base: http://mars-ollama:11434
```

### Defensive Settings Added
```yaml
litellm_settings:
  max_parallel_requests: 10
  num_retries: 3
  callbacks: ["dynamic_rate_limiter_v3"]

router_settings:
  routing_strategy: "simple-shuffle"
  enable_pre_call_checks: false  # Prevent model iteration

general_settings:
  max_parallel_requests: 5
```

## Testing Strategy

### Mock Server Approach
Use a mock HTTP server to definitively count backend calls:

```python
class MockLLMServer:
    """Context manager for mock LLM server that counts requests."""

    def get_request_count(self) -> int:
        return _mock_state.get_request_count()
```

### Test Cases Required

1. **Single Model Request**: 1 LiteLLM request = 1 backend call
2. **Multiple Models Configured**: Only requested model called
3. **Model Alias Resolution**: Alias resolves to single backend call
4. **Rate Limiting**: Parallel requests properly limited

### Ollama-Based Validation
Safe testing using local Ollama (no token consumption):
- Monitor Ollama container logs for request count
- Verify 1:1 mapping between proxy requests and backend calls

## Key Files

- `litellm/router.py` - Core routing logic
- `litellm/router_strategy/simple_shuffle.py` - Shuffle strategy implementation
- `litellm/proxy/route_llm_request.py` - Request routing entry point
- `litellm/proxy/health_check.py` - Health check (iterates all models)

## Recommendations

1. **Add request-level logging** that tracks:
   - Incoming request model
   - All backend calls made for that request
   - Easy audit of 1:1 relationship

2. **Create regression test** that:
   - Configures N models pointing to mock server
   - Makes 1 request
   - Asserts mock received exactly 1 call

3. **Add startup validation** that:
   - Warns if >10 models configured
   - Suggests rate limiting for large model lists

4. **Document model broadcast risk** in:
   - Configuration examples
   - Production deployment guides

## Related Files

- Test: `tests/regression/test_litellm_call_counting.py`
- Test: `tests/regression/test_litellm_single_model_routing.py`
- Lessons Learned: `mars-dev/docs/lessons-learned/2024-12-01-litellm-model-broadcast-token-drain.md`
