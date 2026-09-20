"""
多轮对话封装。

支持 vllm 和 mock 两种引擎；按 session_id 隔离历史。
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
        base_model: Optional[str] = None,
        enable_lora: bool = False,
    ) -> None:
        if use_mock:
            from src.inference.mock_engine import MockEngine

            self.engine = MockEngine(model_path=model_path, base_model=base_model)
        else:
            from src.inference.vllm_engine import VLLMEngine

            self.engine = VLLMEngine(model_path=model_path, enable_lora=enable_lora)

        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt or (
            "你是一个电商客服助手，回答要自然、礼貌、专业，避免敏感内容。"
        )
        self._sessions: dict[str, list[dict]] = {}

    def _new_history(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}]

    def _history(self, session_id: str) -> list[dict]:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._new_history()
        return self._sessions[session_id]

    def reset(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions[session_id] = self._new_history()
        logger.info(f"chat history reset session={session_id or '*'}")

    def _build_prompt(self, history: list[dict]) -> str:
        tokenizer = getattr(self.engine, "tokenizer", None)
        trimmed = [history[0]] + history[1:][-(self.max_history_turns * 2):]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                trimmed,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for m in trimmed:
            role = m["role"]
            if role == "system":
                lines.append(f"System: {m['content']}")
            elif role == "user":
                lines.append(f"User: {m['content']}")
            else:
                lines.append(f"Assistant: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def chat(
        self,
        user_input: str,
        max_tokens: int = 256,
        session_id: str = "default",
    ) -> str:
        history = self._history(session_id)
        history.append({"role": "user", "content": user_input})
        prompt = self._build_prompt(history)
        outputs = self.engine.generate([prompt], max_tokens=max_tokens)
        reply = outputs[0].strip()
        history.append({"role": "assistant", "content": reply})
        return reply

    def stream_chat(self, user_input: str, session_id: str = "default") -> Iterator[str]:
        reply = self.chat(user_input, session_id=session_id)
        for ch in reply:
            yield ch
