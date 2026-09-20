#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m src.training.sft_train \
  --config configs/sft.yaml
