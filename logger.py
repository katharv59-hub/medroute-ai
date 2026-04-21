"""
MedRoute AI — Logging Framework
================================
Centralised logging for the entire system.
- Dual output: terminal (coloured) + logs/system.log (rotating)
- Drop-in replacement for print() via get_logger(name)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ── resolve log dir relative to this file ──────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(_BASE_DIR, "logs")
LOG_FILE   = os.path.join(LOG_DIR, "system.log")

os.makedirs(LOG_DIR, exist_ok=True)

# ── ANSI colour codes for console output ───────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Coloured formatter for the console handler."""
    def format(self, record):
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


_FMT_FILE    = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
_FMT_CONSOLE = "%(asctime)s | %(levelname)s | [%(name)s] %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ── root logger (configured once) ──────────────────────────────────────────
_root = logging.getLogger("medroute")
_root.setLevel(logging.DEBUG)

if not _root.handlers:
    # Rotating file handler — 5 MB × 3 backups
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024,
                               backupCount=3, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATE_FMT))
    _root.addHandler(_fh)

    # Console handler
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_ColourFormatter(_FMT_CONSOLE, datefmt=_DATE_FMT))
    _root.addHandler(_ch)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the medroute namespace."""
    return _root.getChild(name)
