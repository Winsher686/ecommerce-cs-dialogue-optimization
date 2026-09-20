"""
统一配置加载。

用法:
    from src.utils.config import load_config
    cfg = load_config("configs/sft.yaml")
"""

import os
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _override_from_env(cfg)
    logger.info(f"loaded config from {path}")
    return cfg


def _override_from_env(cfg: dict) -> dict:
    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ:
            cfg[key] = _cast(os.environ[env_key], cfg[key])
            logger.info(f"override {key} from env: {cfg[key]}")
    return cfg


def _cast(value: str, ref: Any) -> Any:
    if isinstance(ref, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(ref, int):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    return value


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    logger.info(f"saved config to {path}")


if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/sft.yaml")
    print(cfg)
