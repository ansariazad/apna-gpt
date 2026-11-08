#!/usr/bin/env bash
# ============================================================================
# setup_and_run_mac.sh — One-Command macOS Installer & Runner for All 5 Modules
# ============================================================================
# Reference Sources:
#   1. Apna GPT   : https://www.youtube.com/watch?v=kCc8FmEb1nY
#   2. RAG        : https://www.youtube.com/watch?v=sVcwVQRHIc8
#   3. Agents     : https://www.anthropic.com/engineering/building-effective-agents
#   4. MCP Server : https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server
#   5. AI Evals   : https://hamel.dev/blog/posts/evals/index.html
#
# Usage on your Mac (Terminal):
#   chmod +x setup_and_run_mac.sh
#   ./setup_and_run_mac.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "Setting up Full-Stack AI Engineering Suite on macOS..."
echo "Working directory: $PROJECT_DIR"
echo "========================================================================"

if command -v uv >/dev/null 2>&1; then
  echo "[1/3] Found uv: $(uv --version). Creating virtual environment (.venv)..."
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install torch numpy httpx "mcp[cli]" pytest || true
else
  echo "[1/3] Running with system Python3 ($(python3 --version))."
  echo "      (Optional: install uv from https://astral.sh/uv for full PyTorch MPS & MCP SDK env)"
fi

echo "[2/3] Executing all 5 AI Engineering modules..."
python3 01_apna_gpt.py --steps 150 --sample-tokens 160
python3 02_rag_from_scratch.py
python3 03_effective_agents.py
python3 04_mcp_server.py --self-test
python3 05_ai_evals_suite.py

echo ""
echo "[3/3] Complete! All 5 modules and evaluation traces ran cleanly in: $PROJECT_DIR"
