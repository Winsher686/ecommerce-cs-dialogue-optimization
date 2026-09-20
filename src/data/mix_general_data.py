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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"saved {len(data)} items to {path}")


def to_sft_sample(item: dict) -> list[dict]:
    """
    统一成 SFT 训练格式: {prompt, response, source}。
    兼容 messages 多轮对话。
    """
    if "prompt" in item and "response" in item:
        sample = {
            "prompt": item["prompt"],
            "response": item["response"],
            "source": item.get("source", "unknown"),
        }
        return [sample]

    msgs = item.get("messages") or []
    samples = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        prompt_msgs = msgs[:i]
        if not prompt_msgs or prompt_msgs[-1].get("role") != "user":
            continue
        samples.append({
            "prompt": prompt_msgs,
            "response": m.get("content", ""),
            "source": item.get("source", "general"),
        })
    return samples


def generate_demo_general(n: int = 300, seed: int = 42) -> list[dict]:
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
            "prompt": [{"role": "user", "content": q}],
            "response": a,
            "source": "general",
        })
    return data


def mix(sft: list[dict], general: list[dict], ratio: float) -> list[dict]:
    """
    ratio 表示通用数据占 sft 数据的比例。
    例如 ratio=0.1 表示通用数据量 = len(sft) * 0.1。
    """
    sft_norm = []
    for item in sft:
        sft_norm.extend(to_sft_sample(item))

    general_norm = []
    for item in general:
        general_norm.extend(to_sft_sample(item))

    if not general_norm:
        logger.warning("general data is empty, return sft only")
        return sft_norm

    target_n = int(len(sft_norm) * ratio)
    if target_n <= 0:
        return sft_norm

    if len(general_norm) >= target_n:
        sampled = random.sample(general_norm, target_n)
    else:
        sampled = general_norm * (target_n // len(general_norm) + 1)
        sampled = sampled[:target_n]

    mixed = sft_norm + sampled
    random.shuffle(mixed)
    logger.info(f"sft={len(sft_norm)}, general={len(sampled)}, mixed={len(mixed)}")
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
