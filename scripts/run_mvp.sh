#!/usr/bin/env bash
# 用模拟数据 + 0.5B 小模型走通本地 Demo（不依赖 vLLM / AWQ）
set -e
cd "$(dirname "$0")/.."

python -m src.data.preprocess --generate-demo --demo-size "${DEMO_SIZE:-500}"
python -m src.data.audit --input data/processed/cleaned.jsonl --output docs/data_card.md
python -m src.data.mine_negatives --input data/processed/cleaned.jsonl --no-embedding
python -m src.data.build_dataset
python -m src.data.mix_general_data --generate-demo --demo-size 80

export SFT_CONFIG=configs/mvp/sft.yaml
export DPO_CONFIG=configs/mvp/dpo.yaml
bash scripts/run_sft.sh
bash scripts/run_dpo.sh

export USE_MOCK_ENGINE=1
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export FINETUNED_PATH="${FINETUNED_PATH:-outputs/mvp/dpo_merged}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export FINETUNED_RATIO="${FINETUNED_RATIO:-0.1}"
python -m src.inference.api
