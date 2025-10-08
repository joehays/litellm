#!/bin/bash
# Run functional test with CAPRA credentials

set -e

# Set environment variables
export ASKSAGE_API_KEY="$CAPRA_API_TOKEN"
export ASKSAGE_CA_CERT_PATH="$HOME/dev/joe-docs/dev-ops/Certificates_PKCS7_v5_14_DoD/DoD_PKE_CA_chain.pem"
export PYTHONPATH=.

echo "Testing AskSage provider with CAPRA..."
echo "API Key length: ${#ASKSAGE_API_KEY}"
echo "CA Cert: $ASKSAGE_CA_CERT_PATH"
echo ""

/usr/bin/python3 test_asksage_functional.py
