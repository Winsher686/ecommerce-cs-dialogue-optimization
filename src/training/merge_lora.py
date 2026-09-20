"""
将 LoRA adapter merge 进基座，得到可直接加载的完整权重。

SFT/DPO 默认只保存 adapter；vLLM、AWQ、部分 transformers 路径需要 merge 后的目录。

用法:
    python -m src.training.merge_lora \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/mvp/sft \
        --output outputs/mvp/sft_merged
"""

import argparse
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def merge_peft_model(model, tokenizer, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    if hasattr(model, "merge_and_unload"):
        logger.info("merging LoRA into base weights")
        model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"merged model saved to {output_path}")
    return model


def merge_from_disk(base_path: str, adapter_path: str, output_path: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"loading base: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    logger.info(f"loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    merge_peft_model(model, tokenizer, output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_from_disk(args.base, args.adapter, args.output)
