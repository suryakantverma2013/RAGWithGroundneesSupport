"""
logging_setup.py — shared logging configuration for the Haystack/Milvus scripts
================================================================================
Used by:
    ingest_pdf.py                 → logs/ingest_pdf.log
    haystack_milvus_hybrid_rag.py → logs/hybrid_rag_query.log

Features
--------
* CLI-selectable destination — ``--log-dest file`` (default) writes to a
  size-rotated file under logs/ (5 MB × 3 backups); ``--log-dest console``
  streams to stdout instead.
* CLI-selectable level — ``--log-level INFO`` (default) | DEBUG | WARN.
  DEBUG is the troubleshooting / maximum-verbosity mode.
* Nanosecond timestamps — stdlib ``%(asctime)s`` only reaches milliseconds,
  so a LogRecord factory stamps ``time.time_ns()`` on every record and a
  custom Formatter renders ``HH:MM:SS.nnnnnnnnn``.  (Windows clock
  granularity is ~100 ns; the trailing digits reflect that.)
* Third-party noise capping — docling / RapidOCR / huggingface etc. are
  pinned to WARNING so pipeline progress stays readable even at DEBUG; a few
  loggers whose only WARNING is non-actionable library noise (Haystack's
  evaluator PromptBuilder advisory) are pinned one notch higher, to ERROR.

Ordering constraint
-------------------
Both scripts must configure logging BEFORE their heavy imports so that
import-time messages (Tesseract probe, Langfuse activation) land in the log.
Because the full argparse run happens much later, setup_logging_from_argv()
pre-parses just the two logging flags from sys.argv; the script's full parser
re-declares them via add_logging_args() so they appear in --help.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
LOG_MAX_BYTES = 5 * 1024 * 1024   # rotate at ~5 MB
LOG_BACKUP_COUNT = 3              # keep .log + .log.1/.2/.3

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Third-party libraries are chatty at INFO (and torrential at DEBUG) — cap
# them so our own pipeline progress lines stay readable at every level.
NOISY_LOGGERS = (
    "docling", "docling_core", "docling_ibm_models",
    "RapidOCR", "rapidocr",
    "huggingface_hub", "transformers",
    # httpx + its transport layer + telemetry: wire-level DEBUG chatter that
    # drowns out pipeline troubleshooting signal
    "httpx", "httpcore", "urllib3", "posthog",
)

# Loggers whose only WARNING-level output is non-actionable library noise we
# can't fix from our code — capped one notch higher (to ERROR) than the WARNING
# cap above.  Haystack's LLM evaluators (FaithfulnessEvaluator,
# ContextRelevanceEvaluator) build an internal PromptBuilder WITHOUT
# required_variables and warn "PromptBuilder has N prompt variables, but
# required_variables is not set" on every run.  Our own PromptBuilders already
# set required_variables, so silencing this logger hides only that
# library-internal advisory, not anything from our pipelines.
QUIET_LOGGERS = (
    "haystack.components.builders.prompt_builder",
)

# Warning-message patterns (regex, matched from the start) silenced outright via
# the `warnings` filter — distinct from the loggers above: these are issued by
# the `warnings` module, not `logging`, and are non-actionable from our code.
#   • milvus-haystack still calls the older ORM-style pymilvus API internally
#     (utility.has_collection, Collection, Index.to_dict, …); every store init
#     therefore emits a handful of PyMilvusDeprecationWarnings we can't fix here.
# Matched on message text (not the category class) so this module needn't import
# pymilvus, which would pull heavy deps in before the scripts' own imports.
SUPPRESSED_WARNINGS = (
    r".*ORM-style PyMilvus API.*",
)


# ─────────────────────────────────────────────────────────────────────────────
# Nanosecond timestamps
# LogRecord.created is a time.time() float — sub-microsecond precision is
# already lost by the time a Formatter sees it.  A record factory captures
# time.time_ns() at record creation; the Formatter renders it.
# ─────────────────────────────────────────────────────────────────────────────
_base_record_factory = logging.getLogRecordFactory()


def _ns_record_factory(*args, **kwargs) -> logging.LogRecord:
    record = _base_record_factory(*args, **kwargs)
    record.created_ns = time.time_ns()
    return record


class NanosecondFormatter(logging.Formatter):
    """Renders %(asctime)s as HH:MM:SS.nnnnnnnnn from the record's ns stamp."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ns = getattr(record, "created_ns", None) or int(record.created * 1_000_000_000)
        secs, frac = divmod(ns, 1_000_000_000)
        return f"{time.strftime('%H:%M:%S', time.localtime(secs))}.{frac:09d}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI flags  (shared between the pre-parse below and each script's full parser)
# ─────────────────────────────────────────────────────────────────────────────
def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Declare --log-dest / --log-level on *parser* (shows them in --help)."""
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--log-dest",
        choices=("file", "console"),
        default="file",
        help="Log destination: size-rotated file under logs/ or the console",
    )
    group.add_argument(
        "--log-level",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARN", "WARNING"),
        default="INFO",
        metavar="LEVEL",
        help="Logging verbosity: DEBUG (troubleshooting), INFO, WARN",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(
    log_file_name: str,
    *,
    dest: str = "file",
    level: str = "INFO",
) -> Path | None:
    """
    Configure the root logger for this process.

    Parameters
    ----------
    log_file_name : file name under logs/ (used only when dest == "file")
    dest          : "file" (rotating file under logs/) or "console" (stdout)
    level         : "DEBUG" | "INFO" | "WARN"/"WARNING"

    Returns
    -------
    Path of the active log file, or None when logging to the console.
    """
    logging.setLogRecordFactory(_ns_record_factory)

    formatter = NanosecondFormatter(LOG_FORMAT)
    log_path: Path | None = None
    if dest == "file":
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / log_file_name
        handler: logging.Handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        # Single console breadcrumb so a file-logged run isn't silent.
        # ASCII only — callers may not have reconfigured stdout to UTF-8 yet.
        print(f"[logging] level={level} -> {log_path}")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    for old in root.handlers[:]:  # replace any prior configuration
        root.removeHandler(old)
    root.addHandler(handler)

    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for quiet in QUIET_LOGGERS:
        logging.getLogger(quiet).setLevel(logging.ERROR)

    # Drop known non-actionable warnings (see SUPPRESSED_WARNINGS) BEFORE
    # captureWarnings: an "ignore" filter prevents the warning from being issued
    # at all, so it never reaches the py.warnings logger.  Inserted at the front
    # of the filter list, these take precedence over the default filters.
    for pattern in SUPPRESSED_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)

    # Route the REMAINING Python `warnings` through logging instead of raw
    # stderr: with --log-dest file they land in the log file and the console
    # stays clean.  Anything not matched above is still surfaced, just tidily.
    logging.captureWarnings(True)

    return log_path


def echo_to_console(logger_name: str) -> logging.Logger:
    """
    Return *logger_name*'s logger, additionally wired to stdout — for output
    that must reach the user every run (e.g. the end-of-run summary) even when
    --log-dest file routes everything else to the log file.

    Records still propagate to the root logger, so they land in the log file
    too.  When the root logger already streams to stdout (--log-dest console),
    no extra handler is added — avoiding duplicate console lines.
    """
    logger = logging.getLogger(logger_name)
    streams_stdout = lambda handlers: any(  # noqa: E731
        getattr(h, "stream", None) is sys.stdout for h in handlers
    )
    if not streams_stdout(logging.getLogger().handlers) and not streams_stdout(logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(NanosecondFormatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger


def setup_logging_from_argv(log_file_name: str) -> Path | None:
    """
    Pre-parse just --log-dest / --log-level from sys.argv and configure
    logging.  Call at module top, before heavy imports; the script's full
    parser should re-declare the flags via add_logging_args() for --help.
    """
    pre = argparse.ArgumentParser(add_help=False)
    add_logging_args(pre)
    args, _ = pre.parse_known_args()
    return setup_logging(log_file_name, dest=args.log_dest, level=args.log_level)
