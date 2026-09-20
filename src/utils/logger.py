"""
统一日志模块。

用法:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("hello")
"""

import os
import sys
from datetime import datetime
from loguru import logger as _logger


# 默认日志目录
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名按日期切分
_log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# 控制台格式
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# 文件格式（更详细，便于排查）
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def _init_logger() -> None:
    """初始化 loguru，只执行一次。"""
    # 移除默认 handler
    _logger.remove()

    # 控制台
    _logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=os.environ.get("LOG_LEVEL", "INFO"),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    # 文件
    _logger.add(
        _log_file,
        format=_FILE_FORMAT,
        level="DEBUG",
        rotation="00:00",       # 每天零点切分
        retention="30 days",    # 保留 30 天
        compression="zip",      # 旧日志压缩
        encoding="utf-8",
        enqueue=True,           # 多进程安全
        backtrace=True,
        diagnose=False,
    )


_init_logger()


def get_logger(name: str = "app"):
    """
    获取 logger。

    Args:
        name: 通常传 __name__，方便定位来源。

    Returns:
        loguru.Logger，已绑定 name。
    """
    return _logger.bind(name=name)


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")