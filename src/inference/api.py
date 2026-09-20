"""
FastAPI 推理服务。

支持两种引擎:
    - vllm  : 生产用，vLLM + AWQ
    - mock  : MVP 用，transformers 直接推理

通过环境变量切换:
    USE_MOCK_ENGINE=1  使用 mock 引擎
    默认使用 vllm
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.logger import get_logger
from src.inference.router import GrayRouter
from src.inference.lora_manager import LoRAManager

logger = get_logger(__name__)

app = FastAPI(title="E-commerce CS Dialogue API")

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/awq")
FINETUNED_PATH = os.environ.get("FINETUNED_PATH", "outputs/dpo")
FINETUNED_RATIO = float(os.environ.get("FINETUNED_RATIO", "0.1"))
USE_MOCK = os.environ.get("USE_MOCK_ENGINE", "0") == "1"

# MVP 模式下用小模型路径
if USE_MOCK:
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    FINETUNED_PATH = os.environ.get("FINETUNED_PATH", "outputs/mvp/dpo")

router = GrayRouter(finetuned_ratio=FINETUNED_RATIO)
lora_manager = LoRAManager()
lora_manager.register("dpo", FINETUNED_PATH)

_bot = None
_bot_finetuned = None


def get_bot(finetuned: bool = False):
    global _bot, _bot_finetuned
    from src.inference.chat import ChatBot

    if finetuned:
        if _bot_finetuned is None:
            logger.info(f"lazy init finetuned ChatBot ({'mock' if USE_MOCK else 'vllm'})")
            _bot_finetuned = ChatBot(model_path=FINETUNED_PATH, use_mock=USE_MOCK)
        return _bot_finetuned
    else:
        if _bot is None:
            logger.info(f"lazy init base ChatBot ({'mock' if USE_MOCK else 'vllm'})")
            _bot = ChatBot(model_path=MODEL_PATH, use_mock=USE_MOCK)
        return _bot


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    max_tokens: int = 256


class ChatResponse(BaseModel):
    reply: str
    target: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "mock" if USE_MOCK else "vllm"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        target = router.route()
        finetuned = target == "finetuned"
        bot = get_bot(finetuned=finetuned)
        reply = bot.chat(req.message, max_tokens=req.max_tokens)
        return ChatResponse(reply=reply, target=target, session_id=req.session_id)
    except Exception as e:
        logger.error(f"chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    get_bot(False).reset()
    get_bot(True).reset()
    return {"status": "reset"}


@app.get("/router/status")
def router_status() -> dict:
    return router.status()


@app.post("/router/ratio")
def set_ratio(ratio: float) -> dict:
    router.set_ratio(ratio)
    return router.status()


@app.post("/router/rollback")
def rollback() -> dict:
    router.rollback()
    return router.status()


@app.post("/router/recover")
def recover() -> dict:
    router.recover()
    return router.status()


@app.get("/lora/list")
def list_lora() -> dict:
    return {"adapters": lora_manager.list_adapters()}
