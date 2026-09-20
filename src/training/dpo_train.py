"""
DPO 偏好对齐训练。

关键点:
    - 加载 SFT 后的 LoRA 作为 policy
    - PEFT 场景下 ref_model=None，由 TRL 在 disable adapter 时计算参考对数概率
    - 只对回答部分计算 loss（TRL 按 prompt/chosen/rejected 切分）
    - 自定义 IPO/DPO 公式见 src.training.losses，供实验与单测使用

用法:
    python -m src.training.dpo_train --config configs/dpo.yaml
    python -m src.training.dpo_train --config configs/mvp/dpo.yaml
"""

import argparse
import json
import os

import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.training.merge_lora import is_lora_adapter, merge_peft_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed

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


def build_dataset(path: str, tokenizer) -> Dataset:
    raw = load_jsonl(path)
    samples = []
    for item in raw:
        if "prompt" not in item or "chosen" not in item or "rejected" not in item:
            continue
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        eos = tokenizer.eos_token or ""
        samples.append({
            "prompt": prompt_text,
            "chosen": item["chosen"] + eos,
            "rejected": item["rejected"] + eos,
        })
    if not samples:
        raise ValueError(f"empty DPO dataset: {path}")
    logger.info(f"built DPO dataset with {len(samples)} pairs")
    return Dataset.from_list(samples)


def load_policy(cfg: dict):
    """
    policy: 基座 + SFT LoRA（可继续训练）。
    ref:    PEFT 下传 None，避免再加载一份同等大小的参考模型。
    """
    import torch

    base_path = cfg.get("base_model", cfg["model_name_or_path"])
    sft_path = cfg["model_name_or_path"]
    dtype = resolve_torch_dtype(cfg)
    device_map = resolve_device_map(cfg)

    logger.info(f"loading base model: {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    if is_lora_adapter(sft_path):
        logger.info(f"loading SFT LoRA: {sft_path}")
        policy = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
        policy.print_trainable_parameters()
        return policy

    logger.info(f"SFT path is a full model, use as policy: {sft_path}")
    if os.path.abspath(sft_path) != os.path.abspath(base_path):
        policy = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        return policy
    return base


def _build_dpo_trainer(policy, tokenizer, dataset, dpo_args):
    kwargs = {
        "model": policy,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": dataset,
    }
    try:
        return DPOTrainer(**kwargs, processing_class=tokenizer)
    except TypeError:
        return DPOTrainer(**kwargs, tokenizer=tokenizer)


def main(config_path: str) -> None:
    import torch

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    use_cuda = torch.cuda.is_available()

    base_path = cfg.get("base_model", "Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = load_policy(cfg)
    dataset = build_dataset(
        cfg.get("dataset_path", "data/processed/dpo_dataset.jsonl"),
        tokenizer,
    )

    grad_ckpt = cfg.get("gradient_checkpointing", use_cuda)
    if grad_ckpt and hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    dpo_args = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-5),
        beta=cfg.get("beta", 0.1),
        bf16=bool(cfg.get("bf16", False)) and use_cuda,
        fp16=bool(cfg.get("fp16", False)) and use_cuda,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=grad_ckpt,
        max_length=cfg.get("max_seq_len", 2048),
        max_prompt_length=cfg.get("max_prompt_len", 1024),
        remove_unused_columns=False,
    )

    trainer = _build_dpo_trainer(policy, tokenizer, dataset, dpo_args)

    logger.info("start DPO training")
    trainer.train()

    logger.info(f"saving DPO LoRA to {cfg['output_dir']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    if cfg.get("merge_and_save", False):
        merged_dir = cfg.get("merged_output_dir", os.path.join(cfg["output_dir"], "merged"))
        merge_peft_model(trainer.model, tokenizer, merged_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dpo.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
