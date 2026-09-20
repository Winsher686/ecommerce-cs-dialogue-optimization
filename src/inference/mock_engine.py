"""
MVP 推理引擎：transformers 加载完整模型或 LoRA adapter，不依赖 vLLM。
"""

import os
from typing import Optional

from src.training.merge_lora import is_lora_adapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockEngine:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        load_adapter = is_lora_adapter(model_path)
        tk_path = tokenizer_path or (base_model if load_adapter else None) or model_path
        base_path = base_model or os.environ.get("BASE_MODEL") or tk_path

        logger.info(f"loading tokenizer: {tk_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tk_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_adapter:
            from peft import PeftModel

            logger.info(f"loading base+LoRA: {base_path} + {model_path} on {device}")
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.model = PeftModel.from_pretrained(base, model_path).to(device)
        else:
            logger.info(f"loading model: {model_path} on {device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device)
        self.model.eval()
        logger.info("MockEngine ready")

    def generate(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        max_new = max_tokens or self.max_new_tokens
        results = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature and temperature > 0:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                )
            else:
                gen_kwargs["do_sample"] = False
            with self.torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(text.strip())
        return results
