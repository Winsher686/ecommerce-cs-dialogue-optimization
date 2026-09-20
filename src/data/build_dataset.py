"""
构建 SFT 数据集与 DPO 正负例数据集。

用法:
    python -m src.data.build_dataset \
        --input data/processed/cleaned.jsonl \
        --sft-output data/processed/sft_dataset.jsonl \
        --dpo-output data/processed/dpo_dataset.jsonl
"""

import argparse
import json
import os
import random

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.preprocess import SENSITIVE_WORDS

logger = get_logger(__name__)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"saved {len(data)} items to {path}")


# ------------------------------------------------------------------
# SFT 数据集
# ------------------------------------------------------------------

def build_sft_dataset(data: list[dict]) -> list[dict]:
    """
    把多轮对话展开成多条 SFT 样本。
    每条样本包含 prompt（到当前 user 为止）和 response（当前 assistant 回复）。
    训练时只对 response 计算 loss。
    """
    samples = []
    for d in data:
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prompt_msgs = msgs[:i]
            if not prompt_msgs or prompt_msgs[-1]["role"] != "user":
                continue
            samples.append({
                "prompt": prompt_msgs,
                "response": m["content"],
                "source": d.get("source", "unknown"),
            })
    logger.info(f"sft samples: {len(samples)}")
    return samples


# ------------------------------------------------------------------
# DPO 数据集
# ------------------------------------------------------------------

def _is_good_reply(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    for w in SENSITIVE_WORDS:
        if w in text:
            return False
    return True


def build_dpo_dataset(
    data: list[dict],
    negative_candidates: list[dict] | None = None,
) -> list[dict]:
    """
    构造 DPO 正负例对：
      - 正例：assistant 回复中质量较高的
      - 负例：优先用 mine_negatives 输出的候选；
              否则从同一批数据里随机挑一条低质量回复作为负例
    """
    negatives_pool = []
    if negative_candidates:
        negatives_pool = [c["rejected"] for c in negative_candidates if c.get("rejected")]

    pairs = []
    for d in data:
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            chosen = m["content"]
            if not _is_good_reply(chosen):
                continue

            prompt_msgs = msgs[:i]
            if not prompt_msgs or prompt_msgs[-1]["role"] != "user":
                continue

            # 找负例
            rejected = None
            if negatives_pool:
                rejected = random.choice(negatives_pool)
            else:
                # 从其他对话里随机挑一条低质量回复
                others = [
                    x["content"]
                    for od in data
                    for x in od["messages"]
                    if x["role"] == "assistant" and not _is_good_reply(x["content"])
                ]
                if others:
                    rejected = random.choice(others)

            if not rejected or rejected == chosen:
                continue

            pairs.append({
                "prompt": prompt_msgs,
                "chosen": chosen,
                "rejected": rejected,
                "source": d.get("source", "unknown"),
            })

    logger.info(f"dpo pairs: {len(pairs)}")
    return pairs


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/cleaned.jsonl")
    p.add_argument("--negatives", type=str, default="data/processed/negative_candidates.jsonl")
    p.add_argument("--sft-output", type=str, default="data/processed/sft_dataset.jsonl")
    p.add_argument("--dpo-output", type=str, default="data/processed/dpo_dataset.jsonl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    data = load_jsonl(args.input)
    logger.info(f"loaded {len(data)} dialogues")

    negatives = []
    if os.path.exists(args.negatives):
        negatives = load_jsonl(args.negatives)
        logger.info(f"loaded {len(negatives)} negative candidates")

    sft = build_sft_dataset(data)
    save_jsonl(sft, args.sft_output)

    dpo = build_dpo_dataset(data, negatives)
    save_jsonl(dpo, args.dpo_output)