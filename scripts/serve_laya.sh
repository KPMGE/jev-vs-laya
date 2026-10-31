#!/usr/bin/env bash
# Serve Laya locally on the Mac GPU with Jev's wire protocol at http://127.0.0.1:8000.
# LAYA_HOST must stay on loopback: laya-serve binds 0.0.0.0 by default.
# Fine-tuned checkpoints in models/laya-ft-* are registered automatically under their folder name.
set -euo pipefail
cd "$(dirname "$0")/.."

export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
export LAYA_DEVICE="${LAYA_DEVICE:-mps}"
export LAYA_MODELS="${LAYA_MODELS:-english}"
export LAYA_DEFAULT_MODEL="${LAYA_DEFAULT_MODEL:-english}"
export LAYA_MAX_LOADED="${LAYA_MAX_LOADED:-3}"  # base + two fine-tunes, ~2 GB each in fp32

if [ -z "${LAYA_EXTRA_MODELS:-}" ]; then
  extra=""
  for dir in models/laya-ft-*/; do
    [ -f "${dir}rl_agent_config.json" ] && [ -f "${dir}model.safetensors" ] || continue  # skip half-copied folders
    name="$(basename "$dir")"
    extra="${extra:+$extra,}\"$name\":\"$PWD/${dir%/}\""
  done
  [ -n "$extra" ] && export LAYA_EXTRA_MODELS="{$extra}"
fi
echo "models: $LAYA_MODELS ${LAYA_EXTRA_MODELS:-}"

exec uv run laya-serve
