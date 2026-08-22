"""全局日志配置（loguru）。"""

import sys
from pathlib import Path

from loguru import logger

from src.core.config import settings

logger.remove()

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

logger.add(sys.stderr, level=settings.log_level, format=CONSOLE_FORMAT)

log_dir = Path(settings.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
    level=settings.log_level,
    format=FILE_FORMAT,
)

__all__ = ["logger"]
