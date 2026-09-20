"""数据层单元测试。"""

import json
import os
import tempfile

from src.data.preprocess import (
    clean_messages,
    deduplicate,
    mask_sensitive,
    save_jsonl,
)
from src.data.build_dataset import build_sft_dataset, build_dpo_dataset
from src.data.mix_general_data import mix


def test_mask_sensitive():
    assert "[PHONE]" in mask_sensitive("我的电话是13812345678")
    assert "[EMAIL]" in mask_sensitive("邮箱是abc@example.com")
    assert "[ID]" in mask_sensitive("身份证110101199001011234")


def test_clean_messages_basic():
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ]
    out = clean_messages(msgs)
    assert len(out) == 2
    assert out[0]["role"] == "user"


def test_clean_messages_merge_same_role():
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "您好"},
    ]
    out = clean_messages(msgs)
    assert len(out) == 2
    assert "你好" in out[0]["content"]
    assert "在吗" in out[0]["content"]


def test_clean_messages_drop_leading_assistant():
    msgs = [
        {"role": "assistant", "content": "您好"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ]
    out = clean_messages(msgs)
    assert out[0]["role"] == "user"


def test_clean_messages_too_short():
    msgs = [{"role": "user", "content": "你好"}]
    assert clean_messages(msgs) == []


def test_deduplicate():
    d1 = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    d2 = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    out = deduplicate([d1, d2])
    assert len(out) == 1


def test_build_sft_dataset():
    data = [{
        "messages": [
            {"role": "user", "content": "我的订单在哪"},
            {"role": "assistant", "content": "请提供订单号"},
            {"role": "user", "content": "12345"},
            {"role": "assistant", "content": "已为您查询"},
        ]
    }]
    samples = build_sft_dataset(data)
    assert len(samples) == 2
    assert samples[0]["response"] == "请提供订单号"


def test_build_dpo_dataset():
    data = [{
        "messages": [
            {"role": "user", "content": "我的订单在哪"},
            {"role": "assistant", "content": "请提供订单号，我帮您查询物流"},
        ]
    }]
    negatives = [{"rejected": "不知道。"}]
    pairs = build_dpo_dataset(data, negatives)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "请提供订单号，我帮您查询物流"
    assert pairs[0]["rejected"] == "不知道。"


def test_mix_general_data():
    sft = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]} for i in range(100)]
    general = [{"messages": [{"role": "user", "content": "g"},
                              {"role": "assistant", "content": "g"}]} for _ in range(50)]
    mixed = mix(sft, general, ratio=0.1)
    assert len(mixed) == 110