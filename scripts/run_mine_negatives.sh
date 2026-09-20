#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m src.data.mine_negatives \
  --input  data/processed/cleaned.jsonl \
  --output data/processed/negative_candidates.jsonl \
  --threshold 0.75
