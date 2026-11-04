"""Logging system with secret scrubbing and dual trace outputs."""

import logging
import re
import sys


class SecretScrubbingFormatter(logging.Formatter):
    """Formatter that scrubs API keys and sensitive tokens from log outputs."""

    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
        re.compile(r"gsk_[A-Za-z0-9_-]{20,}"),
        re.compile(r"hf_[A-Za-z0-9_-]{20,}"),
        re.compile(r"bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE),
    ]

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        for pattern in self.SECRET_PATTERNS:
            formatted = pattern.sub("[REDACTED_SECRET]", formatted)
        return formatted


def setup_logger(log_level: str = "INFO", trace_file: str = "peter_trace.log") -> logging.Logger:
    logger = logging.getLogger("peter")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console Handler (INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_fmt = SecretScrubbingFormatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File Trace Handler (DEBUG+)
    try:
        file_handler = logging.FileHandler(trace_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = SecretScrubbingFormatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Could not initialize log file {trace_file}: {e}\n")

    return logger


logger = setup_logger()
