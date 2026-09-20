"""
多轮对话封装。

支持 vllm 和 mock 两种引擎。
"""

from typing import Iterator, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatBot:
    def __init__(
        self,
        model_path: str,
        max_history_turns: int = 10,
        system_prompt: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        if use_mock:
            from src.inference.mock_engine import MockEngine
            self.engine = MockEngine(model_path=model_path)
        else:
            from src.inference.vllm_engine import VLLMEngine
            self.engine = VLLMEngine(model_path=model_path)

        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt or (
            "你是一个电商客服助手，回答要自然、礼貌、专业，避免敏感内容。"
        )
        self.history: list[dict] = []
        self._reset_history()

    def _reset_history(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def reset(self) -> None:
        self._reset_history()
        logger.info("chat history reset")

    def _build_prompt(self) -> str:
        lines = []
        for m in self.history[-self.max_history_turns * 2:]:
            if m["role"] == "system":
                lines.append(f"System: {m['content']}")
            elif m["role"] == "user":
                lines.append(f"User: {m['content']}")
            else:
                lines.append(f"Assistant: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def chat(self, user_input: str, max_tokens: int = 256) -> str:
        self.history.append({"role": "user", "content": user_input})
        prompt = self._build_prompt()
        outputs = self.engine.generate([prompt], max_tokens=max_tokens)
        reply = outputs[0].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def stream_chat(self, user_input: str) -> Iterator[str]:
        reply = self.chat(user_input)
        for ch in reply:
            yield ch
