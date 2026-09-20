"""
按比例把通用指令数据混入 SFT 数据，缓解灾难性遗忘。

用法:
    python -m src.data.mix_general_data \
        --sft data/processed/sft_dataset.jsonl \
        --general data/raw/general_instructions.jsonl \
        --output data/processed/sft_dataset_mixed.jsonl \
        --ratio 0.1
"""

import argparse
import json
import os
import random

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.warning(f"file not found: {path}")
        return []
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


def generate_demo_general(n: int = 300, seed: int = 42) -> list[dict]:
    """生成模拟通用指令数据。"""
    rng = random.Random(seed)
    templates = [
        ("请介绍一下{topic}", "{topic}是一门很有意思的领域，可以从基础概念入手学习。"),
        ("什么是{topic}？", "{topic}指的是一个常见概念，通常用于描述某类现象或方法。"),
        ("如何学习{topic}？", "建议先了解基础，再通过实践加深理解，最后做项目巩固。"),
        ("{topic}有什么优点？", "{topic}的优点包括易用、灵活、社区活跃等。"),
    ]
    topics = ["机器学习", "Python", "数据结构", "操作系统", "网络协议", "数据库", "算法"]
    data = []
    for _ in range(n):
        tpl, ans = rng.choice(templates)
        topic = rng.choice(topics)
        q = tpl.format(topic=topic)
        a = ans.format(topic=topic)
        data.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "source": "general",
        })
    return data


def mix(sft: list[dict], general: list[dict], ratio: float) -> list[dict]:
    """
    ratio 表示通用数据占 sft 数据的比例。
    例如 ratio=0.1 表示通用数据量 = len(sft) * 0.1。
    """
    if not general:
        logger.warning("general data is empty, return sft only")
        return sft

    target_n = int(len(sft) * ratio)
    if target_n <= 0:
        return sft

    if len(general) >= target_n:
        sampled = random.sample(general, target_n)
    else:
        sampled = general * (target_n // len(general) + 1)
        sampled = sampled[:target_n]

    mixed = sft + sampled
    random.shuffle(mixed)
    logger.info(f"sft={len(sft)}, general={len(sampled)}, mixed={len(mixed)}")
    return mixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", type=str, default="data/processed/sft_dataset.jsonl")
    p.add_argument("--general", type=str, default="data/processed/general_instructions.jsonl")
    p.add_argument("--output", type=str, default="data/processed/sft_dataset_mixed.jsonl")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--generate-demo", action="store_true")
    p.add_argument("--demo-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    sft = load_jsonl(args.sft)

    if args.generate_demo:
        general = generate_demo_general(args.demo_size)
        save_jsonl(general, args.general)
    else:
        general = load_jsonl(args.general)

    mixed = mix(sft, general, args.ratio)
    save_jsonl(mixed, args.output)