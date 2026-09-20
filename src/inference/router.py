"""
灰度路由：按流量比例分配请求到基座模型 / 微调模型。

用法:
    from src.inference.router import GrayRouter
    router = GrayRouter(finetuned_ratio=0.1)
    target = router.route()  # 返回 "base" 或 "finetuned"
"""

import random
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RouterConfig:
    finetuned_ratio: float = 0.1   # 10% 走微调
    base_name: str = "base"
    finetuned_name: str = "finetuned"


class GrayRouter:
    def __init__(self, finetuned_ratio: float = 0.1) -> None:
        if not 0.0 <= finetuned_ratio <= 1.0:
            raise ValueError("finetuned_ratio must be in [0, 1]")
        self.config = RouterConfig(finetuned_ratio=finetuned_ratio)
        self._rollback = False

    def route(self) -> str:
        """
        返回目标模型名称。
        回滚状态下强制走 base。
        """
        if self._rollback:
            return self.config.base_name
        if random.random() < self.config.finetuned_ratio:
            return self.config.finetuned_name
        return self.config.base_name

    def set_ratio(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        self.config.finetuned_ratio = ratio
        logger.info(f"finetuned_ratio set to {ratio}")

    def rollback(self) -> None:
        """快速回滚：100% 走基座。"""
        self._rollback = True
        logger.warning("router rollback: 100% base model")

    def recover(self) -> None:
        """恢复灰度。"""
        self._rollback = False
        logger.info("router recovered")

    def status(self) -> dict:
        return {
            "finetuned_ratio": self.config.finetuned_ratio,
            "rollback": self._rollback,
        }