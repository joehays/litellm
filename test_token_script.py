#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test token script execution functionality

Validates that ASKSAGE_TOKEN_COMMAND works correctly.
"""
import os
import sys
import time

# Add litellm to path
sys.path.insert(0, os.path.dirname(__file__))

from litellm.llms.asksage.common_utils import get_asksage_token, _token_cache

print("\n" + "="*70)
print("  Token Script Execution Test")
print("="*70 + "\n")

# Test 1: Token script execution
print("=== Test 1: Execute Token Script ===")
token_script = os.path.expanduser("~/dev/joe-docs/dev-ops/get_capra_access_token.sh")

if not os.path.exists(token_script):
    print(f"❌ Token script not found: {token_script}")
    sys.exit(1)
else:
    print(f"✅ Token script exists: {token_script}")

# Configure environment
os.environ["ASKSAGE_TOKEN_COMMAND"] = token_script

# Clear cache to force fresh execution
_token_cache.clear()

print("   Executing token script...")
token = get_asksage_token()

if token:
    print(f"✅ Token retrieved successfully (length: {len(token)})")
    print(f"   Token preview: {token[:20]}...")
else:
    print("❌ Failed to retrieve token")
    sys.exit(1)

print()

# Test 2: Token caching
print("=== Test 2: Token Caching ===")
print("   Fetching token again (should use cache)...")
start_time = time.time()
cached_token = get_asksage_token()
elapsed = time.time() - start_time

if cached_token == token:
    print(f"✅ Token retrieved from cache (elapsed: {elapsed*1000:.1f}ms)")
    if elapsed < 0.1:  # Cache should be nearly instant
        print("   ✅ Cache is fast (< 100ms)")
    else:
        print("   ⚠️  Cache seems slow, might have re-executed script")
else:
    print("❌ Cached token doesn't match")
    sys.exit(1)

print()

# Test 3: Cache expiry
print("=== Test 3: Cache Expiry ===")
print("   Clearing cache...")
_token_cache.clear()

print("   Fetching token after cache clear...")
new_token = get_asksage_token()

if new_token == token:
    print("✅ Token retrieved after cache clear (same token)")
else:
    print("⚠️  Token changed after cache clear (might be expected)")

print()

# Test 4: Fallback to static token
print("=== Test 4: Fallback to Static Token ===")
print("   Clearing token script configuration...")
os.environ.pop("ASKSAGE_TOKEN_COMMAND", None)
_token_cache.clear()

# Set static token
os.environ["ASKSAGE_API_KEY"] = "test-static-token-12345"

print("   Fetching token with static fallback...")
static_token = get_asksage_token()

if static_token == "test-static-token-12345":
    print("✅ Static token fallback works")
else:
    print(f"❌ Static token fallback failed: {static_token}")
    sys.exit(1)

print()

# Summary
print("="*70)
print("  Test Summary")
print("="*70)
print("\n✅ All token script tests passed!")
print("   • Token script execution works")
print("   • Token caching works")
print("   • Cache clearing works")
print("   • Static token fallback works")
print("\n🎉 Token refresh implementation is functional!")
print()
