"""
日志模块

使用 loguru 替代标准 logging，提供更友好的日志格式和自动文件轮转。
其他模块统一使用: from backend.utils.logger import logger
"""

import os
import sys
from loguru import logger

from backend.config import get_settings


def setup_logger() -> None:
    """
    初始化日志配置。

    - 控制台输出：彩色格式，便于开发调试
    - 文件输出：JSON 格式，便于日志分析，自动按 10MB 轮转
    """
    settings = get_settings()

    # 移除 loguru 默认的 handler，避免重复输出
    logger.remove()

    # 控制台输出 —— 彩色格式，适合开发环境
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出 —— 按大小轮转，保留最近 7 天
    logger.add(
        os.path.join(settings.log_dir, "app_{time:YYYY-MM-DD}.log"),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",       # 单文件最大 10MB
        retention="7 days",     # 保留 7 天
        compression="zip",     # 旧日志自动压缩
        encoding="utf-8",
    )

    logger.info("日志系统初始化完成，日志级别: {}", settings.log_level)


# 模块导入时不自动初始化，由 main.py 启动时调用 setup_logger()
# 其他模块直接 from backend.utils.logger import logger 即可使用
