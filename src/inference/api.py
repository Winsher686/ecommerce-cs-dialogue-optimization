"""
FastAPI 推理服务。

支持两种引擎:
    - vllm  : 生产用，vLLM（路径含 awq 时自动量化）
    - mock  : Demo / MVP，transformers 直接推理

环境变量:
    USE_MOCK_ENGINE=1
    MODEL_PATH / FINETUNED_PATH / FINETUNED_RATIO / BASE_MODEL
    METRICS_PORT  若设置则启动 Prometheus 端口
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.inference.lora_manager import LoRAManager
from src.inference.metrics_server import (
    record_request,
    render_metrics,
    set_finetuned_ratio,
    start_metrics_server,
)
from src.inference.router import GrayRouter
from src.utils.logger import get_logger

logger = get_logger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_ENGINE", "0") == "1"

if USE_MOCK:
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    FINETUNED_PATH = os.environ.get("FINETUNED_PATH", "outputs/mvp/dpo")
else:
    MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/awq")
    FINETUNED_PATH = os.environ.get("FINETUNED_PATH", "outputs/dpo")

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct" if USE_MOCK else "Qwen/Qwen3-8B")
FINETUNED_RATIO = float(os.environ.get("FINETUNED_RATIO", "0.1"))

router = GrayRouter(finetuned_ratio=FINETUNED_RATIO)
lora_manager = LoRAManager()
lora_manager.register("dpo", FINETUNED_PATH)

_bot = None
_bot_finetuned = None


def _local_path_missing(path: str) -> bool:
    if not path:
        return True
    looks_local = (
        path.startswith("outputs")
        or path.startswith(".")
        or os.path.sep in path
        or (len(path) >= 2 and path[1] == ":")
    )
    return looks_local and not os.path.exists(path)


def _finetuned_ready() -> bool:
    adapters = lora_manager.list_adapters()
    return any(a["name"] == "dpo" and a["available"] for a in adapters)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _local_path_missing(FINETUNED_PATH) or not _finetuned_ready():
        logger.warning("finetuned weights not found, force traffic to base")
        router.set_ratio(0.0)
    set_finetuned_ratio(router.config.finetuned_ratio)
    metrics_port = os.environ.get("METRICS_PORT")
    if metrics_port:
        start_metrics_server(int(metrics_port))
    yield


app = FastAPI(title="E-commerce CS Dialogue API", lifespan=lifespan)


def get_bot(finetuned: bool = False):
    global _bot, _bot_finetuned
    from src.inference.chat import ChatBot

    if finetuned:
        if not _finetuned_ready():
            logger.warning("finetuned model unavailable, fallback to base")
            return get_bot(False)
        if _bot_finetuned is None:
            logger.info(f"lazy init finetuned ChatBot ({'mock' if USE_MOCK else 'vllm'})")
            _bot_finetuned = ChatBot(
                model_path=FINETUNED_PATH,
                use_mock=USE_MOCK,
                base_model=BASE_MODEL,
                enable_lora=not USE_MOCK,
            )
            try:
                lora_manager.load("dpo")
            except FileNotFoundError:
                pass
        return _bot_finetuned

    if _bot is None:
        logger.info(f"lazy init base ChatBot ({'mock' if USE_MOCK else 'vllm'})")
        _bot = ChatBot(model_path=MODEL_PATH, use_mock=USE_MOCK, base_model=BASE_MODEL)
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
    return {
        "status": "ok",
        "engine": "mock" if USE_MOCK else "vllm",
        "finetuned_ready": _finetuned_ready(),
        "router": router.status(),
    }


@app.get("/metrics")
def metrics():
    payload, content_type = render_metrics()
    if not payload:
        return Response(content="prometheus_client not installed\n", media_type="text/plain")
    return Response(content=payload, media_type=content_type)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        target = router.route()
        finetuned = target == "finetuned"
        with record_request(target):
            bot = get_bot(finetuned=finetuned)
            reply = bot.chat(
                req.message,
                max_tokens=req.max_tokens,
                session_id=req.session_id or "default",
            )
        return ChatResponse(
            reply=reply,
            target=target,
            session_id=req.session_id or "default",
        )
    except Exception as e:
        logger.error(f"chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    sid = req.session_id or "default"
    if _bot is not None:
        _bot.reset(sid)
    if _bot_finetuned is not None:
        _bot_finetuned.reset(sid)
    return {"status": "reset", "session_id": sid}


@app.get("/router/status")
def router_status() -> dict:
    return router.status()


@app.post("/router/ratio")
def set_ratio(ratio: float) -> dict:
    if not _finetuned_ready() and ratio > 0:
        raise HTTPException(status_code=400, detail="finetuned weights not available")
    router.set_ratio(ratio)
    set_finetuned_ratio(ratio)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
    )
