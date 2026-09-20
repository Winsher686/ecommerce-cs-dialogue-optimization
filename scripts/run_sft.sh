#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m src.training.sft_train --config "${SFT_CONFIG:-configs/sft.yaml}"
