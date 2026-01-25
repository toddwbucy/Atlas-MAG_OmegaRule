"""
Logging configuration for Atlas-MAG.

Provides structured logging with optional file output and
rank-aware formatting for distributed training.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """
    Configure logging for the project.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        rank: Process rank for distributed training
        world_size: Total processes for distributed training
    """
    # Format with rank info if distributed
    if world_size > 1:
        format_str = f"[Rank {rank}/{world_size}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,
    )

    # Reduce noise from other libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
