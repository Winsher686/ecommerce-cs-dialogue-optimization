#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m src.evaluation.evaluate \
  --config configs/inference.yaml \
  --test   data/processed/test.jsonl \
  --output outputs/eval
