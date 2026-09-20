#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m src.training.dpo_train --config "${DPO_CONFIG:-configs/dpo.yaml}"
