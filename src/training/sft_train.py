"""
LoRA SFT 训练。

关键点:
    - LoRA 微调
    - 只对回答部分计算 loss
    - 9:1 混入通用指令数据，防灾难性遗忘
    - warmup + cosine 调度
    - 可选把 LoRA merge 成完整权重，供推理 / 量化使用

用法:
    python -m src.training.sft_train --config configs/sft.yaml
    python -m src.training.sft_train --config configs/mvp/sft.yaml
"""

import argparse
import json
import os

import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.training.merge_lora import merge_peft_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed

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


def resolve_torch_dtype(cfg: dict):
    import torch

    if cfg.get("bf16"):
        return torch.bfloat16
    if cfg.get("fp16"):
        return torch.float16
    return torch.float32


def resolve_device_map(cfg: dict) -> str:
    import torch

    device_map = cfg.get("device_map")
    if device_map:
        return device_map
    return "auto" if torch.cuda.is_available() else "cpu"


def build_prompt_response(tokenizer, prompt_msgs: list[dict], response: str) -> dict:
    """
    用 chat template 拼 prompt 和 response，
    返回 input_ids 和 labels，其中 prompt 部分 label = -100。
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + response + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def build_dataset(path: str, tokenizer, max_len: int = 2048) -> Dataset:
    raw = load_jsonl(path)
    samples = []
    skipped = 0
    for item in raw:
        if "prompt" not in item or "response" not in item:
            skipped += 1
            continue
        enc = build_prompt_response(tokenizer, item["prompt"], item["response"])
        if len(enc["input_ids"]) > max_len:
            enc["input_ids"] = enc["input_ids"][:max_len]
            enc["labels"] = enc["labels"][:max_len]
            enc["attention_mask"] = enc["attention_mask"][:max_len]
        samples.append(enc)
    if skipped:
        logger.warning(f"skipped {skipped} samples without prompt/response")
    if not samples:
        raise ValueError(f"empty SFT dataset: {path}")
    logger.info(f"built dataset with {len(samples)} samples")
    return Dataset.from_list(samples)


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg.get("lora", {})
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.get("r", 8),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        bias="none",
    )


def main(config_path: str) -> None:
    import torch

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    model_path = cfg["model_name_or_path"]
    dataset_path = cfg.get("dataset_path", "data/processed/sft_dataset_mixed.jsonl")
    output_dir = cfg["output_dir"]
    dtype = resolve_torch_dtype(cfg)
    device_map = resolve_device_map(cfg)
    use_cuda = torch.cuda.is_available()

    logger.info(f"loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"loading model: {model_path} dtype={dtype} device_map={device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = build_dataset(dataset_path, tokenizer, cfg.get("max_seq_len", 2048))

    grad_ckpt = cfg.get("gradient_checkpointing", use_cuda)
    if grad_ckpt and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        optim=cfg.get("optim", "adamw_torch"),
        bf16=bool(cfg.get("bf16", False)) and use_cuda,
        fp16=bool(cfg.get("fp16", False)) and use_cuda,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=grad_ckpt,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("start SFT training")
    trainer.train()

    logger.info(f"saving LoRA adapter to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if cfg.get("merge_and_save", False):
        merged_dir = cfg.get("merged_output_dir", os.path.join(output_dir, "merged"))
        merge_peft_model(model, tokenizer, merged_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/sft.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
