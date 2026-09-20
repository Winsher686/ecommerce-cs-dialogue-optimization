"""
vLLM 推理引擎。

特性:
    - vLLM 高吞吐推理
    - AWQ 4-bit 量化加载
    - FlashAttention

用法:
    from src.inference.vllm_engine import VLLMEngine
    engine = VLLMEngine(model_path="outputs/awq")
    outs = engine.generate(["你好"])
"""

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VLLMEngine:
    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = "awq",
        dtype: str = "half",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_flash_attn: bool = True,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError:
            logger.error("请先安装 vllm: pip install vllm")
            raise

        self.model_path = model_path
        self.quantization = quantization

        logger.info(f"init vLLM engine: {model_path}")
        self.llm = LLM(
            model=model_path,
            quantization=quantization,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            enforce_eager=not enable_flash_attn,
            disable_log_stats=False,
        )
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
        """
        vLLM 支持运行时加载 LoRA，这里用 LoRARequest 指定 adapter。
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        lora_req = LoRARequest("dpo_adapter", 1, lora_path)
        outputs = self.llm.generate(prompts, params, lora_request=lora_req)
        return [o.outputs[0].text for o in outputs]