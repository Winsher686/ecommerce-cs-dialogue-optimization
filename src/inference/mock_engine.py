"""
MVP 推理引擎：直接用 transformers 加载小模型，不依赖 vLLM。

用法:
    from src.inference.mock_engine import MockEngine
    engine = MockEngine(model_path="outputs/mvp/dpo")
    outs = engine.generate(["你好"])
"""

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockEngine:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
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

        tk_path = tokenizer_path or model_path
        logger.info(f"loading tokenizer: {tk_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tk_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"loading model: {model_path} on {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
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
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(text.strip())
        return results
