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
- stdout/stderr: NOT attached, ever
- Handler      : buffered, non-blocking PostgresLogHandler mirroring rows
                 into log.sage_kaizen_voice (see main project's
                 log/db/log_schema.sql) — best-effort (degrades to a silent
                 drop if psycopg, the DSN, or the schema/tables are
                 unavailable) — PLUS a small RotatingFileHandler
                 (sage_kaizen_voice.log, 1 MB x 2 backups) re-added
                 2026-07-16 as a crash-safety net: PostgresLogHandler batches
                 records in memory for up to ~2s/200 records before they
                 reach Postgres, and a hard crash (e.g. a BSOD) gives no
                 chance to flush that buffer. This file writes synchronously
                 per log call, independent of the DB batching, so a crash
                 can't lose the data — reconciling it back into Postgres
                 after a real incident is a manual step, not automatic.
                 Deliberately small: its job is to bridge a crash window, not
                 to be a second permanent archive.
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
- Never writes to stdout or stderr (only ever a file, and Postgres).
- Never uses shell redirection.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Format ────────────────────────────────────────────────────────────────── #
_FORMAT       = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FMT     = "%Y-%m-%d %H:%M:%S"
_FALLBACK_REL = "logs/sage_kaizen_voice.log"  # used when paths.yaml is absent

# Crash-safety fallback file sizing (re-added 2026-07-16, same day as the
# DB-only change) — deliberately small: this file's job is to bridge a crash
# window (PostgresLogHandler's in-memory batching), not to be a second
# permanent archive alongside Postgres.
FALLBACK_MAX_BYTES = 1 * 1024 * 1024  # 1 MB
FALLBACK_BACKUP_CNT = 2

# ── Process-level run_id correlation (2026-07-16) ────────────────────────────
# Every LogRecord in this process gets a run_id stamped via a global record
# factory — see sibling projects' sk_logging.py for the full rationale. This
# is a NEW, process-level correlation axis, deliberately separate from the
# existing turn-level ZMQ session_id (see _zmq_handlers.py) — not unifying
# those, just adding a column for log rows.

RUN_ID: str = os.environ.get("SAGE_KAIZEN_RUN_ID") or str(uuid.uuid4())

_prev_record_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _prev_record_factory(*args, **kwargs)
    record.run_id = RUN_ID
    return record


logging.setLogRecordFactory(_record_factory)


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


# ── Shared fallback file handler (module-level singleton, double-checked
# locking) — re-added 2026-07-16 as a crash-safety net alongside
# PostgresLogHandler below; see the module docstring for why. ────────────────

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
            maxBytes=FALLBACK_MAX_BYTES,
            backupCount=FALLBACK_BACKUP_CNT,
            encoding="utf-8",
        )
        h.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
        _handler = h
    return _handler


# ── PostgresLogHandler (2026-07-16) ─────────────────────────────────────────── #
# This project has exactly one log file (sage_kaizen_voice.log -> log.sage_kaizen_voice),
# unlike the sibling projects' multi-file get_logger(file_name=...) pattern, so
# only one PostgresLogHandler instance is ever needed here.

_SOURCE_PROJECT = "sage_kaizen_ai_voice"
_DESCRIPTION_CAP = 65536    # 64 KB — guards against one pathological line bloating a row
_EXCEPTION_CAP = 131072     # 128 KB
_QUEUE_MAXSIZE = 20000
_FLUSH_INTERVAL_S = 2.0
_FLUSH_BATCH_SIZE = 200

# Dedicated internal-diagnostics logger for "DB down"/"DB recovered" notices.
# Must NEVER get a PostgresLogHandler attached (would recurse into the outage
# it's reporting) and must never propagate to a handler-less root logger —
# this project's hard invariant is to never write to stdout/stderr.
_internal_logger = logging.getLogger("sk_logging._internal")
_internal_logger.propagate = False
if not _internal_logger.handlers:
    _internal_logger.setLevel(logging.INFO)
    _internal_dir = _resolve_root() / "logs"
    _internal_dir.mkdir(parents=True, exist_ok=True)
    _internal_handler = RotatingFileHandler(
        filename=str(_internal_dir / "sk_logging_internal.log"),
        maxBytes=1 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    _internal_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    _internal_logger.addHandler(_internal_handler)


class _BoundedQueueHandler(logging.handlers.QueueHandler):
    """
    QueueHandler over a bounded queue.

    Drops the newest record (and counts the drop) instead of blocking the
    calling thread or growing unbounded when the consumer thread falls behind
    a stalled DB. The stock QueueHandler's default unbounded queue would
    otherwise be an unbounded memory leak in the producer process during a
    sustained outage.
    """

    def __init__(self, q: "queue.Queue[logging.LogRecord | None]") -> None:
        super().__init__(q)
        self.dropped = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


class PostgresLogHandler:
    """
    Owns a bounded queue + background consumer thread that batches
    LogRecords into log.sage_kaizen_voice via psycopg3.

    Not a logging.Handler itself — .queue_handler is the actual Handler
    attached to loggers (non-blocking enqueue only); this object owns the
    consumer thread, the DB connection, and the batched INSERT.

    Never raises: any failure (missing psycopg, bad DSN, DB down, missing
    schema) degrades to file-only logging via the sibling RotatingFileHandler,
    with at most one diagnostic notice per state transition.
    """

    def __init__(self, table: str, source_project: str) -> None:
        self.table = table
        self.source_project = source_project
        self._queue: "queue.Queue[logging.LogRecord | None]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self.queue_handler = _BoundedQueueHandler(self._queue)
        self._stop = threading.Event()
        self._conn = None
        self._db_down = False
        self._thread = threading.Thread(
            target=self._run, name=f"pg-log-{table}", daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def _connect(self):
        try:
            import psycopg
        except Exception:
            return None
        try:
            from pg_settings import PgSettings
            dsn = PgSettings().pg_dsn
        except Exception:
            return None
        try:
            return psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        except Exception:
            return None

    def _run(self) -> None:
        batch: list[logging.LogRecord] = []
        while True:
            try:
                record = self._queue.get(timeout=_FLUSH_INTERVAL_S)
            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []
                if self._stop.is_set():
                    break
                continue

            if record is None:  # shutdown sentinel
                if batch:
                    self._flush(batch)
                break

            batch.append(record)
            if len(batch) >= _FLUSH_BATCH_SIZE:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list[logging.LogRecord]) -> None:
        if not batch:
            return
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        if self._conn is None:
            self._note_down()
            return

        rows = []
        for record in batch:
            try:
                rows.append(self._row(record))
            except Exception:
                continue
        if not rows:
            return

        try:
            import psycopg.sql as sql
            with self._conn.cursor() as cur:
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO log.{} "
                        "(log_date, log_type, log_name, description, exception, run_id, source_project) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(self.table)),
                    rows,
                )
            self._note_recovered()
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._note_down()

    def _row(self, record: logging.LogRecord) -> tuple:
        exc_text = None
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))[:_EXCEPTION_CAP]
        return (
            datetime.fromtimestamp(record.created, tz=timezone.utc),
            record.levelname,
            record.name,
            record.getMessage()[:_DESCRIPTION_CAP],
            exc_text,
            getattr(record, "run_id", None),
            self.source_project,
        )

    def _note_down(self) -> None:
        if not self._db_down:
            self._db_down = True
            _internal_logger.warning(
                "PostgresLogHandler(%s): Postgres unreachable — buffering "
                "(bounded, dropped=%d so far) and continuing file-only until recovery.",
                self.table, self.queue_handler.dropped,
            )

    def _note_recovered(self) -> None:
        if self._db_down:
            self._db_down = False
            _internal_logger.info(
                "PostgresLogHandler(%s): Postgres connection recovered.", self.table,
            )

    def close(self) -> None:
        """Drain the queue, flush the final partial batch, and stop the thread."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


_pg_handler_instance: PostgresLogHandler | None = None
_pg_handler_lock = threading.Lock()


def _get_postgres_handler() -> logging.Handler:
    """Return the shared queue-backed Postgres handler, creating it lazily."""
    global _pg_handler_instance
    if _pg_handler_instance is not None:
        return _pg_handler_instance.queue_handler
    with _pg_handler_lock:
        if _pg_handler_instance is None:
            _pg_handler_instance = PostgresLogHandler("sage_kaizen_voice", _SOURCE_PROJECT)
        return _pg_handler_instance.queue_handler


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
        Level INFO, propagate=False, one small RotatingFileHandler (crash
        safety net) plus one (best-effort) buffered PostgresLogHandler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — nothing to do
    logger.addHandler(_get_handler())
    logger.addHandler(_get_postgres_handler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
