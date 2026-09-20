#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${1:-outputs/dpo_merged}"
python -m src.inference.quantize_awq \
  --model  "$MODEL_PATH" \
  --output outputs/awq \
  --bits   4
