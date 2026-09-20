#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m src.inference.quantize_awq \
  --model  outputs/dpo \
  --output outputs/awq \
  --bits   4
