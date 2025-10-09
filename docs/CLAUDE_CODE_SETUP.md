# Claude Code CLI with AskSage Setup Guide

## Overview

This guide provides complete step-by-step instructions for setting up Claude Code CLI (CCC) to use AskSage on a fresh machine. This setup enables you to use AskSage's LLM services through an OpenAI-compatible API for any development project.

## Prerequisites

- Docker installed and running
- Git installed
- AskSage API credentials

## Architecture

```
┌─────────────────┐
│  Claude Code    │  (Your development environment)
│  CLI (CCC)      │
└────────┬────────┘
         │ OpenAI-compatible API
         │ http://localhost:18100
         ▼
┌─────────────────┐
│  LiteLLM Proxy  │  (Running in Docker)
│  Port 18100     │
└────────┬────────┘
         │ AskSage Provider
         │ Bearer Token Auth
         ▼
┌─────────────────┐
│  AskSage API    │  (https://api.asksage.ai/server/)
│  Endpoint       │
└─────────────────┘
```

## Step 1: Clone LiteLLM Repository

```bash
# Clone the LiteLLM repository with AskSage provider
# NOTE: Using forked repo until AskSage provider is merged upstream
cd ~/dev
git clone https://github.com/joehays/litellm.git
cd litellm

# Checkout the AskSage provider branch
git checkout feat/asksage-provider-clean
```

> **Note**: Currently using a forked repository (`joehays/litellm`) because the AskSage provider is pending merge into the official repository. Once the upstream PR is accepted, these instructions will be updated to use `https://github.com/BerriAI/litellm.git` directly.

## Step 2: Prepare Credentials

### 2.1 Get AskSage Access Token Script

You need a script that outputs a fresh Bearer token. Example:

```bash
# Create directory for credentials
mkdir -p ~/credentials

# Create token script (example)
cat > ~/credentials/get_asksage_token.sh << 'EOF'
#!/bin/bash
# Replace with your actual token retrieval method
# This should output ONLY the token to stdout

# Example using API key (adjust for your auth method)
echo "your-asksage-api-key-here"
EOF

chmod +x ~/credentials/get_asksage_token.sh
```

**Note**: Replace the token script content with your actual authentication method.

### 2.2 (Optional) Custom CA Certificates

If you're using a custom endpoint with self-signed certificates:

```bash
# Create directory for certificates
mkdir -p ~/credentials/certs

# Download or copy your CA certificate chain
cp /path/to/your/ca-chain.pem ~/credentials/certs/ca_chain.pem

# Verify certificate
openssl x509 -in ~/credentials/certs/ca_chain.pem -text -noout | head -10
```

### 2.3 Test Token Retrieval

```bash
# Test that token script works
~/credentials/get_asksage_token.sh

# Should output your API key or JWT token
```

## Step 3: Build LiteLLM Docker Image

```bash
cd ~/dev/litellm

# Build the Docker image with AskSage provider
docker build -t litellm-asksage:latest -f docker/Dockerfile.asksage.standalone .

# Verify image was built
docker images | grep litellm-asksage
```

**Expected output:**
```
litellm-asksage   latest   abc123def456   2 minutes ago   500MB
```

## Step 4: Start LiteLLM Proxy

The repository includes pre-configured files for easy deployment:

```bash
cd ~/dev/litellm/docker

# Start the proxy using the pre-made compose file
docker compose -f docker-compose.asksage.yml up -d

# Check logs
docker compose -f docker-compose.asksage.yml logs -f litellm-proxy

# Wait for "Application startup complete" message
# Press Ctrl+C to exit logs
```

**What just happened:**
- Used `docker/docker-compose.asksage.yml` - pre-configured for AskSage
- Used `docker/config.yaml` - includes GPT-4, GPT-3.5, and Claude models
- Mounted your `~/credentials/get_asksage_token.sh` for authentication
- Proxy is now running on port 18100

**Optional: Customize configuration**

If you need to modify settings:

```bash
# Edit the config file
cd ~/dev/litellm/docker
vim config.yaml  # Adjust models, timeouts, etc.

# Edit environment variables
vim docker-compose.asksage.yml  # Change ASKSAGE_API_BASE, etc.

# Restart to apply changes
docker compose -f docker-compose.asksage.yml restart
```

## Step 5: Verify LiteLLM Proxy

```bash
# Test health endpoint
curl http://localhost:18100/health

# Should return JSON with health status

# Test models endpoint
curl http://localhost:18100/v1/models

# Should list available models (gpt-4, gpt-3.5-turbo, claude-3-5-sonnet)
```

## Step 6: Configure Claude Code CLI

### Option 1: Environment Variables (Simplest)

Create a launcher script:

```bash
# Create launcher script
cat > ~/bin/ccc-asksage << 'EOF'
#!/bin/bash
# Claude Code CLI launcher with AskSage/LiteLLM proxy

# Point CCC to local LiteLLM proxy
export OPENAI_API_BASE="http://localhost:18100/v1"
export OPENAI_API_KEY="dummy-key"  # Proxy has no auth

# Launch Claude Code CLI
exec claude-code "$@"
EOF

chmod +x ~/bin/ccc-asksage

# Add to PATH if needed
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: CCC Configuration File (If Supported)

Check if CCC supports custom endpoint configuration:

```bash
# Look for CCC config file
ls ~/.config/claude/

# If config.json exists, edit it:
# Add or modify these fields:
{
  "api": {
    "base_url": "http://localhost:18100/v1",
    "api_key": "dummy-key"
  }
}
```

**Note**: CCC configuration format may vary. Check CCC documentation for exact format.

## Step 7: Test Claude Code CLI with AskSage

```bash
# Create a test directory
mkdir -p ~/test-project
cd ~/test-project

# Launch CCC with AskSage
ccc-asksage

# Or if using regular CCC with env vars:
OPENAI_API_BASE="http://localhost:18100/v1" \
OPENAI_API_KEY="dummy-key" \
claude-code
```

**Test prompt:**
```
List the files in the current directory
```

If successful, CCC will use AskSage through the LiteLLM proxy!

## Step 8: Verify AskSage Usage

```bash
# Check proxy logs to confirm AskSage is being used
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml logs litellm-proxy | tail -50

# Look for log entries showing:
# - Token refresh from script
# - Requests to https://api.asksage.ai/server/
# - Successful responses from AskSage
```

## Daily Usage

### Starting the Proxy

```bash
# Start LiteLLM proxy
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml up -d

# Verify it's running
docker compose -f docker-compose.asksage.yml ps
```

### Using CCC

```bash
# Method 1: Use launcher script
cd ~/your-project
ccc-asksage

# Method 2: Use environment variables
cd ~/your-project
OPENAI_API_BASE="http://localhost:18100/v1" \
OPENAI_API_KEY="dummy-key" \
claude-code
```

### Stopping the Proxy

```bash
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml down
```

### Checking Status

```bash
# Check if proxy is running
docker ps | grep litellm-asksage-proxy

# Check health
curl http://localhost:18100/health

# View logs
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml logs -f
```

## Troubleshooting

### Proxy Won't Start

```bash
# Check Docker logs
docker logs litellm-asksage-proxy

# Common issues:
# 1. Port 18100 already in use
docker ps | grep 18100

# 2. Token script not executable
chmod +x ~/credentials/get_asksage_token.sh

# 3. Certificate path wrong (if using custom certs)
ls -la ~/credentials/certs/ca_chain.pem
```

### Authentication Errors

```bash
# Test token script manually
~/credentials/get_asksage_token.sh

# Should output your API key or token

# Test token inside container
docker exec litellm-asksage-proxy /app/tokens/get_token.sh
```

### Certificate Errors

```bash
# Verify certificate is valid (if using custom certs)
openssl x509 -in ~/credentials/certs/ca_chain.pem -text -noout

# Check certificate inside container
docker exec litellm-asksage-proxy ls -la /app/certs/
docker exec litellm-asksage-proxy cat /app/certs/ca_chain.pem | head -5
```

### CCC Not Using Proxy

```bash
# Verify environment variables are set
echo $OPENAI_API_BASE
echo $OPENAI_API_KEY

# Test proxy manually with curl
curl -X POST http://localhost:18100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

### Token Expiration

```bash
# If your tokens have an expiration time
# LiteLLM automatically refreshes using the token script

# To manually refresh:
~/credentials/get_asksage_token.sh

# Check token age in logs:
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml logs | grep token
```

## Advanced Configuration

### Multiple Endpoints

If you have multiple AskSage endpoints (dev, staging, prod):

```yaml
# Edit ~/dev/litellm/docker/config.yaml
model_list:
  - model_name: gpt-4-dev
    litellm_params:
      model: asksage/gpt-4
      api_base: https://dev.api.asksage.ai/server/

  - model_name: gpt-4-prod
    litellm_params:
      model: asksage/gpt-4
      api_base: https://api.asksage.ai/server/
```

### Custom Model Aliases

```yaml
# Edit ~/dev/litellm/docker/config.yaml
model_list:
  # Alias for quick access
  - model_name: fast
    litellm_params:
      model: asksage/gpt-3.5-turbo
      api_base: ${ASKSAGE_API_BASE}

  - model_name: smart
    litellm_params:
      model: asksage/gpt-4
      api_base: ${ASKSAGE_API_BASE}
```

### Enable Authentication (Optional)

For shared proxy setups:

```yaml
general_settings:
  master_key: "your-secret-key-here"  # Use strong random key
```

Then in CCC:
```bash
export OPENAI_API_KEY="your-secret-key-here"
```

## Security Best Practices

1. **Never commit credentials**
   ```bash
   # Add to .gitignore in any project
   echo "credentials/" >> ~/.gitignore_global
   echo "*.pem" >> ~/.gitignore_global
   echo "*token*" >> ~/.gitignore_global
   ```

2. **Restrict file permissions**
   ```bash
   chmod 600 ~/credentials/get_asksage_token.sh
   chmod 600 ~/credentials/certs/*.pem
   ```

3. **Use separate credentials per environment**
   - Development: dev-specific credentials
   - Production: prod-specific credentials

4. **Rotate tokens regularly**
   - Token script automatically gets fresh tokens
   - Monitor for expired tokens in logs

5. **Keep Docker image updated**
   ```bash
   cd ~/dev/litellm
   git pull origin feat/asksage-provider-clean  # Update to 'main' after upstream PR merge
   docker build -t litellm-asksage:latest -f docker/Dockerfile.asksage.standalone .
   ```

## Summary

### Quick Reference

**Start proxy:**
```bash
cd ~/dev/litellm/docker && docker compose -f docker-compose.asksage.yml up -d
```

**Launch CCC:**
```bash
ccc-asksage
```

**Check status:**
```bash
curl http://localhost:18100/health
```

**View logs:**
```bash
docker logs -f litellm-asksage-proxy
```

**Stop proxy:**
```bash
cd ~/dev/litellm/docker && docker compose -f docker-compose.asksage.yml down
```

### File Locations

- **LiteLLM Repo**: `~/dev/litellm`
- **Proxy Config**: `~/dev/litellm/docker/config.yaml`
- **Docker Compose**: `~/dev/litellm/docker/docker-compose.asksage.yml`
- **Credentials**: `~/credentials/`
- **Launcher Script**: `~/bin/ccc-asksage`

### Ports

- **LiteLLM Proxy**: `http://localhost:18100`
- **Health Check**: `http://localhost:18100/health`
- **Models API**: `http://localhost:18100/v1/models`

---

## Appendix: Complete Fresh Machine Setup Script

```bash
#!/bin/bash
# Complete setup script for Claude Code CLI with AskSage

set -e  # Exit on error

echo "🚀 Setting up Claude Code CLI with AskSage..."

# Step 1: Clone LiteLLM
echo "📦 Cloning LiteLLM repository..."
mkdir -p ~/dev
cd ~/dev
git clone https://github.com/joehays/litellm.git
cd litellm
git checkout feat/asksage-provider-clean

# Step 2: Build Docker image
echo "🐳 Building Docker image..."
docker build -t litellm-asksage:latest -f docker/Dockerfile.asksage.standalone .

# Step 3: Create directories
echo "📁 Creating configuration directories..."
mkdir -p ~/credentials/certs
mkdir -p ~/bin

# Step 4: Prompt for credentials setup
echo ""
echo "⚠️  MANUAL STEPS REQUIRED:"
echo "1. Place your token script at: ~/credentials/get_asksage_token.sh"
echo "2. (Optional) Place CA certificates at: ~/credentials/certs/ca_chain.pem"
echo ""
read -p "Press Enter when credentials are in place..."

# Step 5: Verify credentials
if [[ ! -x ~/credentials/get_asksage_token.sh ]]; then
    echo "❌ Token script not found or not executable"
    exit 1
fi

# Step 6: Create launcher script
cat > ~/bin/ccc-asksage << 'EOF'
#!/bin/bash
export OPENAI_API_BASE="http://localhost:18100/v1"
export OPENAI_API_KEY="dummy-key"
exec claude-code "$@"
EOF

chmod +x ~/bin/ccc-asksage

# Step 7: Update PATH
if ! grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

# Step 8: Start proxy
echo "🚀 Starting LiteLLM proxy..."
cd ~/dev/litellm/docker
docker compose -f docker-compose.asksage.yml up -d

# Step 9: Wait for health check
echo "⏳ Waiting for proxy to be ready..."
sleep 10

# Step 10: Verify
echo "✅ Verifying setup..."
if curl -f http://localhost:18100/health > /dev/null 2>&1; then
    echo "✅ Setup complete!"
    echo ""
    echo "🎉 You can now use Claude Code CLI with AskSage:"
    echo "   ccc-asksage"
    echo ""
    echo "📊 Check status: curl http://localhost:18100/health"
    echo "📋 View logs: docker logs -f litellm-asksage-proxy"
else
    echo "❌ Proxy health check failed. Check logs:"
    echo "   docker logs litellm-asksage-proxy"
    exit 1
fi
```

Save this script as `setup-ccc-asksage.sh`, make it executable, and run it for automated setup.

---

*Last Updated: October 2024*
*Version: 1.0*
