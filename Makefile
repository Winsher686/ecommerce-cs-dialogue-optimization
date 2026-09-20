.PHONY: help data audit mine sft dpo eval quantize api test lint clean

help:
@echo "make data      - 生成模拟数据并清洗"
@echo "make audit     - 数据审计"
@echo "make mine      - 负例挖掘"
@echo "make build     - 构建 SFT/DPO 数据集"
@echo "make sft       - SFT 训练"
@echo "make dpo       - DPO 训练"
@echo "make eval      - 评估"
@echo "make quantize  - AWQ 量化"
@echo "make api       - 启动 API"
@echo "make test      - 单元测试"
@echo "make clean     - 清理输出"

data:
python -m src.data.preprocess --generate-demo --demo-size 2000

audit:
python -m src.data.audit --input data/processed/cleaned.jsonl --output docs/data_card.md

mine:
python -m src.data.mine_negatives --input data/processed/cleaned.jsonl --no-embedding

build:
python -m src.data.build_dataset

sft:
bash scripts/run_sft.sh

dpo:
bash scripts/run_dpo.sh

eval:
bash scripts/run_eval.sh

quantize:
bash scripts/run_quantize.sh

api:
bash scripts/run_api.sh

test:
pytest tests/ -v

lint:
python -m compileall src

clean:
rm -rf outputs/sft/* outputs/dpo/* outputs/eval/*
