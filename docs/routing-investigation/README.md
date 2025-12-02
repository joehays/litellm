# LiteLLM Routing Investigation

This directory contains documentation from investigating a critical token drain incident where LiteLLM routed requests to ALL configured models instead of the single requested model.

## Contents

- **[MODEL_BROADCAST_BUG.md](MODEL_BROADCAST_BUG.md)** - Detailed incident report and investigation findings
- **[SAFE_TESTING_GUIDE.md](SAFE_TESTING_GUIDE.md)** - Guide for validating routing behavior without token consumption

## Related Test Files

Tests to validate routing behavior are located in the MARS repository:

- `mars-dev/tests/regression/test_litellm_single_model_routing.py` - Routing validation tests
- `mars-dev/tests/regression/test_litellm_call_counting.py` - Backend call counting tests

## Quick Summary

### The Problem
6 HTTP requests caused 3,842 backend API calls (38x amplification), draining 4.9M tokens in ~60-90 minutes.

### Key Findings
1. Router code (`simple_shuffle`, `get_available_deployment`) is correct
2. Broadcast occurs elsewhere in the stack (suspected: provider init/callbacks)
3. Config-level mitigations (rate limiting, parallel request limits) are effective

### Validation
With Ollama-only configuration:
- 11 tests pass validating single-model-per-request behavior
- Mock server approach provides definitive backend call counting
- Rate limiting properly enforced

### Prevention
1. Disable AskSage models until root cause identified
2. Use Ollama for development (no token consumption)
3. Add rate limiting and parallel request limits
4. Run regression tests before re-enabling external APIs
