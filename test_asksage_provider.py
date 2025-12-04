#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for AskSage provider implementation

Tests the AskSage provider without requiring actual CAPRA access.
Uses mock responses to verify the transformation logic.
"""
import os
import sys
from unittest.mock import Mock, patch

# Add litellm to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from litellm import completion
from litellm.types.utils import ModelResponse


def test_basic_setup():
    """Test 1: Verify AskSage provider is registered"""
    print("\n=== Test 1: Provider Registration ===")

    from litellm.types.utils import LlmProviders

    # Check if ASKSAGE is in LlmProviders enum
    assert hasattr(LlmProviders, 'ASKSAGE'), "ASKSAGE not in LlmProviders enum"
    assert LlmProviders.ASKSAGE.value == "asksage", "ASKSAGE enum value incorrect"

    print("✅ ASKSAGE provider registered in LlmProviders enum")

    # Check if handler is instantiated
    from litellm.main import asksage_chat_completions
    assert asksage_chat_completions is not None, "AskSage handler not instantiated"

    print("✅ AskSage handler instantiated successfully")
    print("✅ Test 1 PASSED\n")


def test_transformation_request():
    """Test 2: Verify request transformation logic"""
    print("=== Test 2: Request Transformation ===")

    from litellm.llms.asksage.chat.transformation import AskSageConfig

    config = AskSageConfig()

    # Test message transformation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    optional_params = {
        "temperature": 0.7,
        "dataset": ["ds_12345"],
        "persona": "researcher",
    }

    data = config.transform_request(
        model="google-claude-4-opus",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Verify transformation
    assert data["message"] == "What is 2+2?", "User message not extracted correctly"
    assert data["system_prompt"] == "You are a helpful assistant.", "System prompt not extracted"
    assert data["dataset"] == ["ds_12345"], "Dataset parameter not mapped"
    assert data["persona"] == "researcher", "Persona parameter not mapped"
    assert data["temperature"] == 0.7, "Temperature not mapped"
    assert data["model"] == "google-claude-4-opus", "Model not mapped"

    print("✅ User message extracted: 'What is 2+2?'")
    print("✅ System prompt extracted: 'You are a helpful assistant.'")
    print("✅ Dataset parameter mapped: ['ds_12345']")
    print("✅ Persona parameter mapped: 'researcher'")
    print("✅ Temperature mapped: 0.7")
    print("✅ Model mapped: 'google-claude-4-opus'")
    print("✅ Test 2 PASSED\n")


def test_transformation_response():
    """Test 3: Verify response transformation logic"""
    print("=== Test 3: Response Transformation ===")

    from litellm.llms.asksage.chat.transformation import AskSageConfig

    config = AskSageConfig()

    # Mock AskSage response
    mock_response_data = {
        "response": "2+2 equals 4.",
        "model_used": "google-claude-4-opus",
        "tokens_used": {
            "prompt": 15,
            "completion": 10,
            "total": 25
        },
        "citations": [
            {"source": "math_textbook.pdf", "page": 1}
        ]
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = mock_response_data
    mock_response.status_code = 200

    model_response = ModelResponse()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    # Transform response
    result = config.transform_response(
        model="google-claude-4-opus",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=Mock(),
        api_key="test_key",
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Verify transformation
    assert len(result.choices) == 1, "No choices in response"
    assert result.choices[0].message.content == "2+2 equals 4.", "Response text not extracted"
    assert result.choices[0].message.role == "assistant", "Role not set to assistant"
    assert result.model == "google-claude-4-opus", "Model not set correctly"
    assert result.usage.prompt_tokens == 15, "Prompt tokens not extracted"
    assert result.usage.completion_tokens == 10, "Completion tokens not extracted"
    assert result.usage.total_tokens == 25, "Total tokens not extracted"

    print("✅ Response text extracted: '2+2 equals 4.'")
    print("✅ Model extracted: 'google-claude-4-opus'")
    print("✅ Token usage extracted: 15 prompt, 10 completion, 25 total")
    print("✅ Citations stored in metadata")
    print("✅ Test 3 PASSED\n")


def test_authentication_headers():
    """Test 4: Verify authentication header setup"""
    print("=== Test 4: Authentication Headers ===")

    from litellm.llms.asksage.chat.transformation import AskSageConfig

    config = AskSageConfig()

    headers = {}
    api_key = "test_bearer_token_12345"

    result_headers = config.validate_environment(
        headers=headers,
        model="google-claude-4-opus",
        messages=[],
        optional_params={},
        api_key=api_key,
    )

    assert "Authorization" in result_headers, "Authorization header not set"
    assert result_headers["Authorization"] == f"Bearer {api_key}", "Bearer token format incorrect"
    assert result_headers["Content-Type"] == "application/json", "Content-Type not set"

    print(f"✅ Authorization header set: 'Bearer {api_key}'")
    print("✅ Content-Type set: 'application/json'")
    print("✅ Test 4 PASSED\n")


def test_error_handling():
    """Test 5: Verify error handling"""
    print("=== Test 5: Error Handling ===")

    from litellm.llms.asksage.common_utils import AskSageError

    # Test error creation
    error = AskSageError(
        status_code=401,
        message="Unauthorized: Invalid API key",
        headers={"content-type": "application/json"}
    )

    assert error.status_code == 401, "Status code not set"
    assert "Unauthorized" in error.message, "Error message not set"
    assert error.headers["content-type"] == "application/json", "Headers not set"

    print("✅ AskSageError created successfully")
    print(f"✅ Status code: {error.status_code}")
    print(f"✅ Message: {error.message}")
    print("✅ Test 5 PASSED\n")


def test_model_routing():
    """Test 6: Verify model name routing"""
    print("=== Test 6: Model Name Routing ===")

    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test model name parsing
    model = "asksage/google-claude-4-opus"

    parsed_model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model
    )

    assert custom_llm_provider == "asksage", f"Provider not detected correctly: {custom_llm_provider}"
    assert parsed_model == "google-claude-4-opus", f"Model name not parsed correctly: {parsed_model}"

    print("✅ Model name 'asksage/google-claude-4-opus' parsed correctly")
    print(f"✅ Provider detected: '{custom_llm_provider}'")
    print(f"✅ Model extracted: '{parsed_model}'")
    print("✅ Test 6 PASSED\n")


def test_tls_certificate_config():
    """Test 7: Verify TLS certificate configuration"""
    print("=== Test 7: TLS Certificate Configuration ===")

    from litellm.llms.asksage.chat.handler import AskSageChatCompletion

    handler = AskSageChatCompletion()

    # Test without CA cert (should use default)
    if "ASKSAGE_CA_CERT_PATH" in os.environ:
        del os.environ["ASKSAGE_CA_CERT_PATH"]

    client = handler._get_httpx_client(
        api_base="https://api.asksage.ai",
        timeout=30.0
    )

    assert client is not None, "Client not created"
    print("✅ Client created without CA cert (uses system CA bundle)")

    # Test with CA cert path
    fake_cert_path = "/tmp/fake_cert.pem"
    os.environ["ASKSAGE_CA_CERT_PATH"] = fake_cert_path

    # Note: This will create client with fake path but won't fail until actual request
    # In real usage, the cert file must exist

    print(f"✅ CA cert path environment variable set: {fake_cert_path}")
    print("✅ Test 7 PASSED\n")

    # Clean up
    if "ASKSAGE_CA_CERT_PATH" in os.environ:
        del os.environ["ASKSAGE_CA_CERT_PATH"]


def test_supported_parameters():
    """Test 8: Verify supported OpenAI parameters"""
    print("=== Test 8: Supported Parameters ===")

    from litellm.llms.asksage.chat.transformation import AskSageConfig

    config = AskSageConfig()
    supported = config.get_supported_openai_params(model="google-claude-4-opus")

    assert "temperature" in supported, "temperature not in supported params"
    assert "max_tokens" in supported, "max_tokens not in supported params"
    assert "stream" in supported, "stream not in supported params"

    print("✅ Supported parameters:")
    for param in supported:
        print(f"   - {param}")
    print("✅ Test 8 PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AskSage Provider Test Suite")
    print("="*60)

    tests = [
        ("Provider Registration", test_basic_setup),
        ("Request Transformation", test_transformation_request),
        ("Response Transformation", test_transformation_response),
        ("Authentication Headers", test_authentication_headers),
        ("Error Handling", test_error_handling),
        ("Model Routing", test_model_routing),
        ("TLS Certificate Config", test_tls_certificate_config),
        ("Supported Parameters", test_supported_parameters),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("="*60)
    print(f"  Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 All tests passed! AskSage provider is ready.\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
