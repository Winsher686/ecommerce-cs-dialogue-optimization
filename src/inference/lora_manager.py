"""
LoRA 适配器热加载管理器。

注册本地 adapter 目录；vLLM 路径下由引擎的 LoRARequest 真正加载。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.training.merge_lora import is_lora_adapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoRAAdapter:
    name: str
    path: str
    loaded: bool = False
    available: bool = False
    meta: dict = field(default_factory=dict)


class LoRAManager:
    def __init__(self) -> None:
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._active: Optional[str] = None

    def register(self, name: str, path: str, **meta) -> None:
        available = os.path.isdir(path) and (
            is_lora_adapter(path) or os.path.isfile(os.path.join(path, "config.json"))
        )
        if not available:
            logger.warning(f"adapter path not ready: {path}")
        self._adapters[name] = LoRAAdapter(
            name=name,
            path=path,
            available=available,
            meta=meta,
        )
        logger.info(f"registered adapter: {name} -> {path} available={available}")

    def load(self, name: str) -> LoRAAdapter:
        if name not in self._adapters:
            raise KeyError(f"adapter not registered: {name}")
        adapter = self._adapters[name]
        if not adapter.available:
            raise FileNotFoundError(f"adapter not found on disk: {adapter.path}")
        adapter.loaded = True
        self._active = name
        logger.info(f"loaded adapter: {name}")
        return adapter

    def unload(self, name: str) -> None:
        if name in self._adapters:
            self._adapters[name].loaded = False
            if self._active == name:
                self._active = None
            logger.info(f"unloaded adapter: {name}")

    def get_active(self) -> Optional[LoRAAdapter]:
        if self._active is None:
            return None
        return self._adapters[self._active]

    def list_adapters(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "path": a.path,
                "loaded": a.loaded,
                "available": a.available,
                "meta": a.meta,
            }
            for a in self._adapters.values()
        ]

    def switch(self, name: str) -> LoRAAdapter:
        return self.load(name)
