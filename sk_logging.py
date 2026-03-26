"""
sk_logging.py — Sage Kaizen AI Voice — Centralized logging
===========================================================
Drop-in import for all submodules in this service.

Usage
-----
    from sk_logging import get_logger
    _LOG = get_logger("sage_kaizen.voice.pipeline")

Properties
----------
- Level        : INFO
- propagate    : False — messages never bubble to the root logger
- stdout/stderr: NOT attached — all output goes to the rotating file only
- Handler      : single RotatingFileHandler shared across all loggers
                 Max 5 MB per file · 5 backups (sage_kaizen_voice.log.1 … .5)
- Encoding     : UTF-8
- Format       : %(asctime)s | %(levelname)s | %(name)s | %(message)s
- Date format  : %Y-%m-%d %H:%M:%S

Log file resolution (first match wins)
---------------------------------------
1. ``logs.default`` key in ``config/paths.yaml`` (relative to project root)
2. Hard-coded Python fallback: ``logs/sage_kaizen_voice.log``

Project root resolution
-----------------------
``$SAGE_KAIZEN_ROOT`` env var → ``Path(__file__).resolve().parent``

Hard invariants
---------------
- get_logger() is idempotent: safe to call at module import time.
- All log directories are created automatically on first call.
- Never writes to stdout or stderr.
- Never uses shell redirection.
"""

from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Format ──────────────────────────────────────────────────────────────────
_FORMAT       = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FMT     = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES    = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_CNT   = 5                # keeps .log.1 … .log.5
_FALLBACK_REL = "logs/sage_kaizen_voice.log"  # used when paths.yaml is absent


# ── Project root ─────────────────────────────────────────────────────────────

def _resolve_root() -> Path:
    """Return the project root as an absolute Path."""
    env = os.environ.get("SAGE_KAIZEN_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent


# ── Log file path ─────────────────────────────────────────────────────────────

def _resolve_log_file(root: Path) -> Path:
    """
    Read ``logs.default`` from ``config/paths.yaml``.
    Falls back to ``_FALLBACK_REL`` if the file is missing or unreadable.
    Returns an absolute Path; parent directory is NOT created here.
    """
    rel: str = _FALLBACK_REL
    yaml_path = root / "config" / "paths.yaml"
    if yaml_path.exists():
        try:
            import yaml  # PyYAML — listed in requirements.txt
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            rel = data.get("logs", {}).get("default", _FALLBACK_REL)
        except Exception:
            pass  # any parse/import error → use fallback silently
    return (root / rel).resolve()


# ── Shared handler (module-level singleton, double-checked locking) ──────────

_handler: RotatingFileHandler | None = None
_handler_lock = threading.Lock()


def _get_handler() -> RotatingFileHandler:
    """
    Return the shared RotatingFileHandler, creating it on the first call.
    Thread-safe via double-checked locking.
    """
    global _handler
    if _handler is not None:
        return _handler
    with _handler_lock:
        if _handler is not None:  # re-check inside the lock
            return _handler
        root     = _resolve_root()
        log_file = _resolve_log_file(root)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(
            log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_CNT,
            encoding="utf-8",
        )
        h.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
        _handler = h
    return _handler


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a ``logging.Logger`` keyed by *name*.

    Idempotent: if handlers are already attached the existing logger is
    returned unchanged — no duplicate handlers are ever added.

    Parameters
    ----------
    name:
        Dotted logger name, e.g. ``"sage_kaizen.voice.pipeline"``.

    Returns
    -------
    logging.Logger
        Level INFO, propagate=False, one RotatingFileHandler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — nothing to do
    logger.addHandler(_get_handler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
