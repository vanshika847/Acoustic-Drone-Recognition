"""
Project logger.
"""

import logging
from pathlib import Path


def setup_logger(log_name="processing.log"):

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / log_name

    logger = logging.getLogger(log_name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger