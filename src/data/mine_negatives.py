"""
自动负例挖掘。

思路:
  1. 规则粗筛：长度、重复、模板化、敏感词
  2. Embedding 相似度：低质量回复与高质量回复的相似度低于阈值
  3. 输出负例候选，供人工复核

用法:
    python -m src.data.mine_negatives \
        --input data/processed/cleaned.jsonl \
        --output data/processed/negative_candidates.jsonl \
        --threshold 0.75
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Optional

from src.utils.logger import get_logger
from src.data.preprocess import SENSITIVE_WORDS

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 规则粗筛
# ------------------------------------------------------------------

BAD_PATTERNS = [
    r"^不知道",
    r"^你自己",
    r"^不归我们",
    r"^随便",
    r"^等着",
    r"^呵呵",
]

TEMPLATE_REPLIES = [
    "好的",
    "收到",
    "谢谢",
    "请稍等",
    "亲亲",
]


def rule_filter_one(text: str) -> Optional[str]:
    """返回命中原因，未命中返回 None。"""
    if not text or not text.strip():
        return "empty"
    if len(text.strip()) < 5:
        return "too_short"
    if len(text) > 500:
        return "too_long"
    for p in BAD_PATTERNS:
        if re.search(p, text.strip()):
            return "bad_pattern"
    for w in SENSITIVE_WORDS:
        if w in text:
            return "sensitive"
    if text.strip() in TEMPLATE_REPLIES:
        return "template"
    return None


def rule_filter(data: list[dict]) -> list[dict]:
    """对每条对话的 assistant 回复做规则筛选，命中则标记为负例候选。"""
    candidates = []
    for d in data:
        for i, m in enumerate(d["messages"]):
            if m["role"] != "assistant":
                continue
            reason = rule_filter_one(m["content"])
            if reason:
                candidates.append({
                    "messages": d["messages"][:i + 1],
                    "rejected": m["content"],
                    "reason": f"rule:{reason}",
                })
    logger.info(f"rule filter hit: {len(candidates)}")
    return candidates


# ------------------------------------------------------------------
# Embedding 相似度筛
# ------------------------------------------------------------------

def _load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """延迟加载，避免无网络时报错。"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.warning(f"embedder load failed: {e}")
        return None


def embedding_filter(
    data: list[dict],
    good_replies: list[str],
    threshold: float = 0.75,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    """
    计算每条 assistant 回复与高质量回复集合的最大相似度，
    低于阈值则视为低质量负例候选。
    """
    model = _load_embedder(model_name)
    if model is None:
        logger.warning("skip embedding filter (no model)")
        return []

    import numpy as np

    good_emb = model.encode(good_replies, normalize_embeddings=True)

    candidates = []
    for d in data:
        for i, m in enumerate(d["messages"]):
            if m["role"] != "assistant":
                continue
            text = m["content"]
            if not text.strip():
                continue
            emb = model.encode([text], normalize_embeddings=True)[0]
            sim = float(np.max(good_emb @ emb))
            if sim < threshold:
                candidates.append({
                    "messages": d["messages"][:i + 1],
                    "rejected": text,
                    "reason": f"embedding:{sim:.3f}",
                })
    logger.info(f"embedding filter hit: {len(candidates)}")
    return candidates


def collect_good_replies(data: list[dict], max_n: int = 500) -> list[str]:
    """从数据里挑一批较长的 assistant 回复作为高质量参考。"""
    replies = []
    for d in data:
        for m in d["messages"]:
            if m["role"] == "assistant" and 10 <= len(m["content"]) <= 300:
                replies.append(m["content"])
    replies = list(dict.fromkeys(replies))
    return replies[:max_n]


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def merge_candidates(rule_cands: list[dict], emb_cands: list[dict]) -> list[dict]:
    """合并去重，保留原因。"""
    seen = set()
    out = []
    for c in rule_cands + emb_cands:
        key = c["rejected"]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"saved {len(data)} candidates to {path}")


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/cleaned.jsonl")
    p.add_argument("--output", type=str, default="data/processed/negative_candidates.jsonl")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--no-embedding", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_jsonl(args.input)
    logger.info(f"loaded {len(data)} dialogues")

    rule_cands = rule_filter(data)

    if args.no_embedding:
        emb_cands = []
    else:
        good = collect_good_replies(data)
        emb_cands = embedding_filter(data, good, args.threshold)

    merged = merge_candidates(rule_cands, emb_cands)

    reason_counter = Counter(c["reason"].split(":")[0] for c in merged)
    logger.info(f"reason distribution: {dict(reason_counter)}")

    save_jsonl(merged, args.output)