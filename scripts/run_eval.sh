#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m src.evaluation.evaluate \
  --config "${EVAL_CONFIG:-configs/inference.yaml}" \
  --test   "${TEST_PATH:-data/processed/test.jsonl}" \
  --output outputs/eval
