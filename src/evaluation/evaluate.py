"""
评估入口。

用法:
    python -m src.evaluation.evaluate \
        --config configs/inference.yaml \
        --test   data/processed/test.jsonl \
        --output outputs/eval
"""

import argparse
import json
import os

import yaml

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_all

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"saved report to {path}")


def build_prompts(test_data: list[dict], tokenizer) -> list[str]:
    prompts = []
    for item in test_data:
        prompt_msgs = item.get("prompt")
        if prompt_msgs is None and "messages" in item:
            msgs = item["messages"]
            for i, m in enumerate(msgs):
                if m["role"] == "assistant":
                    prompt_msgs = msgs[:i]
                    break
        if not prompt_msgs:
            continue
        prompts.append(
            tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def collect_refs(test_data: list[dict]) -> tuple[list[str], list[str]]:
    """从测试集里取参考回复（chosen / rejected 或 messages 里的 assistant）。"""
    refs, rejected = [], []
    for item in test_data:
        if "chosen" in item:
            refs.append(item["chosen"])
            rejected.append(item.get("rejected", ""))
        elif "messages" in item:
            asst = [m["content"] for m in item["messages"] if m["role"] == "assistant"]
            refs.append(asst[-1] if asst else "")
            rejected.append("")
    return refs, rejected


def generate_with_vllm(model_path: str, prompts: list[str], cfg: dict) -> list[str]:
    from src.inference.vllm_engine import VLLMEngine

    engine = VLLMEngine(model_path=model_path)
    return engine.generate(
        prompts,
        max_tokens=cfg.get("max_new_tokens", 512),
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 0.9),
    )


def evaluate(config_path: str, test_path: str, output_dir: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    test_data = load_jsonl(test_path)
    logger.info(f"loaded {len(test_data)} test samples")

    # 这里仅用 tokenizer 做 prompt 拼接，不加载完整模型
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer_path", "Qwen/Qwen3-8B"),
        trust_remote_code=True,
    )

    prompts = build_prompts(test_data, tokenizer)
    refs, rejected = collect_refs(test_data)

    model_path = cfg["model_name_or_path"]
    logger.info(f"generating with {model_path}")
    preds = generate_with_vllm(model_path, prompts, cfg)

    report = compute_all(preds, refs, rejected if any(rejected) else None)
    report["num_samples"] = len(preds)
    report["model_path"] = model_path

    os.makedirs(output_dir, exist_ok=True)
    save_json(report, os.path.join(output_dir, "report.json"))

    with open(os.path.join(output_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for p, r in zip(preds, refs):
            f.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")

    logger.info(f"report: {report}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/inference.yaml")
    p.add_argument("--test", type=str, default="data/processed/test.jsonl")
    p.add_argument("--output", type=str, default="outputs/eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.test, args.output)