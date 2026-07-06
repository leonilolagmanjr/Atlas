"""Atlas logging setup.

Centralizes Python logging configuration so all modules can emit consistent
structured logs for startup, indexing, search, and error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO, log_to_file: Optional[str] = None) -> None:
    """Configure global logging.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_to_file: Optional file path for a log file handler.
    """

    root = logging.getLogger()
    if root.handlers:
        # Avoid double-configuring in interactive runs.
        return

    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_to_file:
        Path(log_to_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_to_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

