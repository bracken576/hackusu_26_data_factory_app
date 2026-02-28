"""Append-only audit trail — every query and chat event must be logged here."""
import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "audit_trail.csv"

_FIELDS = [
    "timestamp", "session_id", "user_email", "user_role",
    "action_type", "ai_source", "source_tables",
    "query_text", "row_count", "execution_time_ms", "pii_accessed", "error",
]

_session_id = str(uuid.uuid4())  # one ID per app process


def _ensure_log_file() -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    if not _LOG_PATH.exists():
        with open(_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def log_event(
    action_type: str,
    user_email: str = "unknown",
    user_role: str = "viewer",
    ai_source: str | None = None,
    source_tables: str = "",
    query_text: str = "",
    row_count: int = 0,
    execution_time_ms: int = 0,
    pii_accessed: bool = False,
    error: str = "",
) -> None:
    """Append one event to the audit trail. Non-blocking — errors are logged but not raised."""
    try:
        _ensure_log_file()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": _session_id,
            "user_email": user_email,
            "user_role": user_role,
            "action_type": action_type,
            "ai_source": ai_source or "",
            "source_tables": source_tables,
            "query_text": query_text[:500],  # cap length
            "row_count": row_count,
            "execution_time_ms": execution_time_ms,
            "pii_accessed": pii_accessed,
            "error": error[:200],
        }
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writerow(row)
    except Exception as exc:
        logger.error("audit_service.log_event failed: %s", exc)


def read_audit_log(limit: int = 500) -> "pd.DataFrame":
    import pandas as pd
    _ensure_log_file()
    try:
        df = pd.read_csv(_LOG_PATH)
        return df.tail(limit).iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=_FIELDS)
