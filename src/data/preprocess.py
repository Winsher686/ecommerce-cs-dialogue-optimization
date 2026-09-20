"""
数据预处理：读取原始客服日志，清洗后输出统一多轮对话格式。

用法:
    # 生成模拟数据并清洗
    python -m src.data.preprocess --generate-demo --demo-size 2000

    # 清洗真实数据
    python -m src.data.preprocess --input data/raw/logs.jsonl --output data/processed/cleaned.jsonl
"""

import argparse
import json
import os
import random
import re
from typing import Iterable

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 敏感信息脱敏
# ------------------------------------------------------------------

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"\d{17}[\dXx]")
ORDER_RE = re.compile(r"(订单号|订单)[:：]?\s*([A-Za-z0-9]{6,})")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SENSITIVE_WORDS = ["傻", "垃圾", "滚", "投诉你", "骗子"]


def mask_sensitive(text: str) -> str:
    """对手机号、身份证、订单号、邮箱做脱敏。"""
    if not text:
        return text
    text = PHONE_RE.sub("[PHONE]", text)
    text = ID_CARD_RE.sub("[ID]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ORDER_RE.sub(lambda m: f"{m.group(1)}[ORDER]", text)
    return text


# ------------------------------------------------------------------
# 基础清洗
# ------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_messages(messages: list[dict]) -> list[dict]:
    """
    清洗单条多轮对话。
    规则：
      - 只保留 role in {user, assistant}
      - 去空 content
      - 相邻同 role 合并
      - 必须以 user 开头
      - 至少一轮 user + assistant
    """
    if not messages:
        return []

    cleaned = []
    for m in messages:
        role = m.get("role")
        content = _normalize_text(m.get("content", ""))
        if role not in ("user", "assistant"):
            continue
        if not content:
            continue
        content = mask_sensitive(content)

        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += " " + content
        else:
            cleaned.append({"role": role, "content": content})

    # 去掉开头的 assistant
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    # 去掉结尾落单的 user
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()

    if len(cleaned) < 2:
        return []

    return cleaned


def _dialogue_key(messages: list[dict]) -> str:
    """用于去重的 key。"""
    return "||".join(f"{m['role']}:{m['content']}" for m in messages)


def deduplicate(data: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in data:
        msgs = item.get("messages", [])
        key = _dialogue_key(msgs)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def filter_low_quality(data: list[dict], min_turns: int = 1, max_turns: int = 30) -> list[dict]:
    """过滤轮数过少或过多的对话。"""
    out = []
    for item in data:
        msgs = item.get("messages", [])
        turns = sum(1 for m in msgs if m["role"] == "user")
        if turns < min_turns or turns > max_turns:
            continue
        out.append(item)
    return out


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------

def load_raw(path: str) -> list[dict]:
    """
    读取原始数据，支持 jsonl 和 json。
    期望每条记录含 messages 字段，或 user/assistant 交替字段。
    """
    if not os.path.exists(path):
        logger.warning(f"input not found: {path}")
        return []

    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"unsupported file format: {path}")

    logger.info(f"loaded {len(items)} raw items from {path}")
    return items


def save_jsonl(data: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"saved {n} items to {path}")


# ------------------------------------------------------------------
# 模拟数据生成器
# ------------------------------------------------------------------

_USER_QUERIES = [
    "我的订单什么时候到？",
    "快递三天没更新了，是不是丢了？",
    "我要退货，怎么操作？",
    "这个商品有优惠吗？",
    "我买错尺码了，能换吗？",
    "你们客服怎么这么慢？",
    "退款什么时候到账？",
    "这个和另一个比哪个好？",
    "发票怎么开？",
    "能不能便宜点？",
    "我刚下的单能改地址吗？",
    "收到的货有破损怎么办？",
    "这个支持七天无理由吗？",
    "有没有优惠券？",
    "物流显示签收但我没收到",
]

_ASSISTANT_REPLIES = [
    "您好，请提供订单号，我帮您查询物流进度。",
    "非常抱歉给您带来不便，我这边帮您催一下快递。",
    "退货可以在订单页点击申请退货，选择原因后会有快递上门取件。",
    "目前该商品有满减活动，下单自动抵扣。",
    "可以换尺码，请告诉我订单号和需要的尺码。",
    "抱歉让您久等了，请问具体遇到什么问题？",
    "退款一般 1-3 个工作日到账，具体以银行处理为准。",
    "两款各有优势，请问您更看重哪方面？我可以帮您对比。",
    "发票可以在订单完成后申请，支持电子发票。",
    "价格已经是最优惠了，您可以关注店铺活动。",
    "未发货前可以修改地址，请提供新地址。",
    "破损请拍照上传，我们核实后为您补发或退款。",
    "支持七天无理由，商品需保持完好。",
    "您可以领取店铺优惠券，下单时自动抵扣。",
    "请提供订单号，我帮您核实签收情况。",
]

_BAD_REPLIES = [
    "不知道。",
    "你自己看订单。",
    "这个不归我们管。",
    "等着吧。",
    "随便你。",
]


def _rand_dialogue(rng: random.Random, n_turns: int) -> dict:
    messages = []
    for _ in range(n_turns):
        messages.append({"role": "user", "content": rng.choice(_USER_QUERIES)})
        messages.append({"role": "assistant", "content": rng.choice(_ASSISTANT_REPLIES)})
    return {"messages": messages, "source": "demo"}


def generate_demo_data(n: int = 2000, seed: int = 42) -> list[dict]:
    """
    生成模拟多轮客服对话，覆盖：
      - 模糊意图
      - 多意图冲突
      - 口语化表达
      - 少量低质量回复
    """
    rng = random.Random(seed)
    data = []
    for i in range(n):
        n_turns = rng.choice([1, 1, 2, 2, 3, 4, 5])
        d = _rand_dialogue(rng, n_turns)

        # 10% 概率插入一条低质量回复，供负例挖掘使用
        if rng.random() < 0.1:
            d["messages"].append({"role": "user", "content": "你怎么这个态度？"})
            d["messages"].append({"role": "assistant", "content": rng.choice(_BAD_REPLIES)})

        # 5% 概率插入口语化表达
        if rng.random() < 0.05:
            d["messages"].insert(
                0, {"role": "user", "content": rng.choice(["在吗", "喂喂喂", "客服？？", "有人吗"])}
            )
            d["messages"].insert(
                1, {"role": "assistant", "content": "您好，我在的，请问有什么可以帮您？"}
            )

        data.append(d)
    logger.info(f"generated {len(data)} demo dialogues")
    return data


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def run(input_path: str | None, output_path: str, demo: bool, demo_size: int) -> None:
    if demo:
        raw_items = generate_demo_data(demo_size)
    else:
        raw_items = load_raw(input_path)

    cleaned = []
    for item in raw_items:
        msgs = item.get("messages")
        if not msgs:
            continue
        c = clean_messages(msgs)
        if c:
            cleaned.append({"messages": c, "source": item.get("source", "unknown")})

    logger.info(f"after clean: {len(cleaned)}")
    cleaned = deduplicate(cleaned)
    logger.info(f"after dedup: {len(cleaned)}")
    cleaned = filter_low_quality(cleaned)
    logger.info(f"after turn filter: {len(cleaned)}")

    save_jsonl(cleaned, output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="原始数据路径")
    p.add_argument("--output", type=str, default="data/processed/cleaned.jsonl")
    p.add_argument("--generate-demo", action="store_true")
    p.add_argument("--demo-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run(args.input, args.output, args.generate_demo, args.demo_size)
