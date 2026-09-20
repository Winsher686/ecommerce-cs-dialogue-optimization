"""
数据审计：统计对话轮数、长度分布、角色比例、敏感词命中，输出报告。

用法:
    python -m src.data.audit --input data/processed/cleaned.jsonl --output docs/data_card.md
"""

import argparse
import json
import os
from collections import Counter
from statistics import mean, median

from src.utils.logger import get_logger
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


def count_turns(data: list[dict]) -> dict:
    turns = [sum(1 for m in d["messages"] if m["role"] == "user") for d in data]
    return {
        "total_dialogues": len(data),
        "total_turns": sum(turns),
        "avg_turns": round(mean(turns), 2) if turns else 0,
        "median_turns": median(turns) if turns else 0,
        "max_turns": max(turns) if turns else 0,
        "min_turns": min(turns) if turns else 0,
        "turn_distribution": dict(Counter(turns)),
    }


def count_lengths(data: list[dict]) -> dict:
    user_lens, asst_lens = [], []
    for d in data:
        for m in d["messages"]:
            n = len(m["content"])
            if m["role"] == "user":
                user_lens.append(n)
            else:
                asst_lens.append(n)
    return {
        "user_avg_len": round(mean(user_lens), 2) if user_lens else 0,
        "user_max_len": max(user_lens) if user_lens else 0,
        "assistant_avg_len": round(mean(asst_lens), 2) if asst_lens else 0,
        "assistant_max_len": max(asst_lens) if asst_lens else 0,
    }


def count_sensitive(data: list[dict]) -> dict:
    hits = Counter()
    for d in data:
        for m in d["messages"]:
            for w in SENSITIVE_WORDS:
                if w in m["content"]:
                    hits[w] += 1
    return dict(hits)


def check_quality_issues(data: list[dict]) -> dict:
    empty_reply = 0
    too_short_reply = 0
    repeated_reply = 0
    for d in data:
        asst = [m["content"] for m in d["messages"] if m["role"] == "assistant"]
        for a in asst:
            if not a.strip():
                empty_reply += 1
            elif len(a) < 5:
                too_short_reply += 1
        if len(asst) != len(set(asst)):
            repeated_reply += 1
    return {
        "empty_reply": empty_reply,
        "too_short_reply": too_short_reply,
        "dialogue_with_repeated_reply": repeated_reply,
    }


def generate_report(data: list[dict], out_path: str) -> None:
    turns = count_turns(data)
    lengths = count_lengths(data)
    sensitive = count_sensitive(data)
    quality = check_quality_issues(data)

    lines = []
    lines.append("# 数据卡（Data Card）\n")
    lines.append("## 概览\n")
    lines.append(f"- 对话总数：{turns['total_dialogues']}")
    lines.append(f"- 总轮数：{turns['total_turns']}")
    lines.append(f"- 平均轮数：{turns['avg_turns']}")
    lines.append(f"- 中位轮数：{turns['median_turns']}")
    lines.append(f"- 最大/最小轮数：{turns['max_turns']} / {turns['min_turns']}\n")

    lines.append("## 轮数分布\n")
    for k, v in sorted(turns["turn_distribution"].items()):
        lines.append(f"- {k} 轮：{v} 条")
    lines.append("")

    lines.append("## 文本长度\n")
    lines.append(f"- 用户平均长度：{lengths['user_avg_len']}")
    lines.append(f"- 用户最大长度：{lengths['user_max_len']}")
    lines.append(f"- 客服平均长度：{lengths['assistant_avg_len']}")
    lines.append(f"- 客服最大长度：{lengths['assistant_max_len']}\n")

    lines.append("## 敏感词命中\n")
    if sensitive:
        for w, c in sensitive.items():
            lines.append(f"- {w}：{c} 次")
    else:
        lines.append("- 无\n")
    lines.append("")

    lines.append("## 质量问题\n")
    for k, v in quality.items():
        lines.append(f"- {k}：{v}")
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"report saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/cleaned.jsonl")
    p.add_argument("--output", type=str, default="docs/data_card.md")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_jsonl(args.input)
    logger.info(f"loaded {len(data)} dialogues")
    generate_report(data, args.output)