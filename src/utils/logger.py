"""
Standardized logging utility for console and file output across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import LOG_FILE_PATH


def get_logger(name: str, log_file: Optional[Path] = LOG_FILE_PATH) -> logging.Logger:
    """
    Create and configure a logger that writes to both stdout and a log file.

    Args:
        name: Name of the module requesting the logger (e.g., __name__).
        log_file: Path to the output log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger