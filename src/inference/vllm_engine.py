"""
vLLM 推理引擎。

特性:
    - vLLM 高吞吐推理
    - 路径名含 awq 时自动开 4-bit
    - 可选 LoRA（enable_lora）

用法:
    from src.inference.vllm_engine import VLLMEngine
    engine = VLLMEngine(model_path="outputs/awq")
    outs = engine.generate(["你好"])
"""

import os
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _default_quantization(model_path: str, quantization: Optional[str]) -> Optional[str]:
    if quantization is not None:
        return quantization or None
    lowered = model_path.replace("\\", "/").lower()
    if "awq" in lowered.split("/")[-1] or lowered.rstrip("/").endswith("/awq"):
        return "awq"
    return None


class VLLMEngine:
    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = None,
        dtype: str = "half",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_flash_attn: bool = True,
        enable_lora: bool = False,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError:
            logger.error("请先安装 vllm: pip install vllm")
            raise

        self.model_path = model_path
        self.quantization = _default_quantization(model_path, quantization)
        self.enable_lora = enable_lora or os.environ.get("VLLM_ENABLE_LORA", "0") == "1"

        logger.info(
            f"init vLLM engine: {model_path} quant={self.quantization} lora={self.enable_lora}"
        )
        llm_kwargs = {
            "model": model_path,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "enforce_eager": not enable_flash_attn,
            "disable_log_stats": False,
            "enable_lora": self.enable_lora,
        }
        if self.quantization:
            llm_kwargs["quantization"] = self.quantization

        self.llm = LLM(**llm_kwargs)
        logger.info("vLLM engine ready")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def generate_with_lora(
        self,
        prompts: list[str],
        lora_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[str]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if not self.enable_lora:
            raise RuntimeError("vLLM engine was created with enable_lora=False")

        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        lora_req = LoRARequest("dpo_adapter", 1, lora_path)
        outputs = self.llm.generate(prompts, params, lora_request=lora_req)
        return [o.outputs[0].text for o in outputs]
