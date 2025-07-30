"""
Logging utilities for training.
"""
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(out: Path, name: str = "rgflow"):
    """
    Set up logging to both file and console.
    
    Args:
        out: Output directory for log files
        name: Logger name (default: "rgflow")
    
    Returns:
        Configured logger instance
    """
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    date = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_h = logging.FileHandler(out / f"{name}_{ts}.log")
    file_h.setFormatter(logging.Formatter(fmt, date))
    cons_h = logging.StreamHandler()
    cons_h.setFormatter(logging.Formatter(fmt, date))

    if not logger.handlers:
        logger.addHandler(file_h)
        logger.addHandler(cons_h)
    return logger