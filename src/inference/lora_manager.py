"""
LoRA 适配器热加载管理器。

支持:
    - 注册多个 adapter
    - 运行时加载 / 卸载
    - 查询当前可用 adapter

用法:
    from src.inference.lora_manager import LoRAManager
    mgr = LoRAManager()
    mgr.register("dpo", "outputs/dpo")
    mgr.load("dpo")
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoRAAdapter:
    name: str
    path: str
    loaded: bool = False
    meta: dict = field(default_factory=dict)


class LoRAManager:
    def __init__(self) -> None:
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._active: Optional[str] = None

    def register(self, name: str, path: str, **meta) -> None:
        if not os.path.exists(path):
            logger.warning(f"adapter path not found: {path}")
        self._adapters[name] = LoRAAdapter(name=name, path=path, meta=meta)
        logger.info(f"registered adapter: {name} -> {path}")

    def load(self, name: str) -> LoRAAdapter:
        if name not in self._adapters:
            raise KeyError(f"adapter not registered: {name}")
        adapter = self._adapters[name]
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
            {"name": a.name, "path": a.path, "loaded": a.loaded, "meta": a.meta}
            for a in self._adapters.values()
        ]

    def switch(self, name: str) -> LoRAAdapter:
        """切换到指定 adapter，等价于 load。"""
        return self.load(name)