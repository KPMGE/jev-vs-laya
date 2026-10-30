#!/usr/bin/env bash
# Serve Laya locally on the Mac GPU with Jev's wire protocol at http://127.0.0.1:8000.
# LAYA_HOST must stay on loopback: laya-serve binds 0.0.0.0 by default.
# Extra env vars pass through, e.g. LAYA_EXTRA_MODELS='{"ours":"models/laya-ft"}' for the fine-tuned model.
set -euo pipefail
cd "$(dirname "$0")/.."

export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
export LAYA_DEVICE="${LAYA_DEVICE:-mps}"
export LAYA_MODELS="${LAYA_MODELS:-english}"
export LAYA_DEFAULT_MODEL="${LAYA_DEFAULT_MODEL:-english}"

exec uv run laya-serve
