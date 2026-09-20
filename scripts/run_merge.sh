#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python -m src.training.merge_lora \
  --base "${BASE_MODEL:-Qwen/Qwen3-8B}" \
  --adapter "${1:-outputs/dpo}" \
  --output "${2:-outputs/dpo_merged}"
