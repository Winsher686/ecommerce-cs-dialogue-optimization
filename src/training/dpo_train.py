"""
DPO 偏好对齐训练。

关键点:
    - 加载 SFT 后的 LoRA 模型作为 policy
    - 复制一份冻结作为 reference
    - 只对回答部分计算 loss
    - 基于 trl.DPOTrainer，可注入自定义 loss

用法:
    python -m src.training.dpo_train --config configs/dpo.yaml
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.training.losses import dpo_loss, compute_logps

logger = get_logger(__name__)


IGNORE_INDEX = -100


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


def build_dataset(path: str, tokenizer, max_len: int = 2048) -> Dataset:
    """
    把 {prompt, chosen, rejected} 展开成 prompt / chosen / rejected 文本，
    交给 DPOTrainer 处理。
    """
    raw = load_jsonl(path)
    samples = []
    for item in raw:
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        samples.append({
            "prompt": prompt_text,
            "chosen": item["chosen"] + tokenizer.eos_token,
            "rejected": item["rejected"] + tokenizer.eos_token,
        })
    logger.info(f"built DPO dataset with {len(samples)} pairs")
    return Dataset.from_list(samples)


def load_policy_and_ref(cfg: dict, tokenizer):
    """
    policy: 加载 SFT 后的 LoRA
    ref:    加载 SFT 后的 LoRA 并冻结
    """
    base_path = cfg.get("base_model", "Qwen/Qwen3-8B")
    sft_path = cfg["model_name_or_path"]

    logger.info(f"loading base model: {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    logger.info(f"loading SFT LoRA: {sft_path}")
    policy = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    policy.print_trainable_parameters()

    # reference：再加载一份 base + LoRA，冻结
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_base, sft_path, is_trainable=False)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    return policy, ref_model


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    base_path = cfg.get("base_model", "Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy, ref_model = load_policy_and_ref(cfg, tokenizer)

    dataset = build_dataset(
        cfg.get("dataset_path", "data/processed/dpo_dataset.jsonl"),
        tokenizer,
        cfg.get("max_seq_len", 2048),
    )

    dpo_args = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-5),
        beta=cfg.get("beta", 0.1),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        max_length=cfg.get("max_seq_len", 2048),
        max_prompt_length=cfg.get("max_prompt_len", 1024),
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("start DPO training")
    trainer.train()

    logger.info(f"saving DPO LoRA to {cfg['output_dir']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dpo.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)