"""日志配置（loguru）：由应用启动时显式装配，不再依赖全局配置。"""

import sys
from pathlib import Path

from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


def configure_logger(
    level: str = "INFO",
    log_dir: str | Path | None = None,
) -> None:
    """（重）配置全局 logger：控制台必开，文件输出可选。"""
    logger.remove()
    logger.add(sys.stderr, level=level, format=CONSOLE_FORMAT)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "{time:YYYY-MM-DD}.log",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            enqueue=True,
            level=level,
            format=FILE_FORMAT,
        )


# 默认仅控制台输出，避免任何模块在应用显式装配前导入即崩溃
configure_logger()

__all__ = ["logger", "configure_logger"]
