"""
Postgres task store for production deployment.
Used by Streamlit (enqueue, poll status) and Worker (update status).
When DATABASE_URL is not set or USE_TASK_QUEUE is false, task queue is disabled.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

# Task types and statuses
TASK_TYPES = (
    "download_results",
    "download_scenarios",
    "run_eval_dirs",
    "generate_summary_csv",
    "build_parquet",
)
TASK_STATUSES = ("pending", "running", "completed", "failed")


def get_database_url() -> Optional[str]:
    """Return DATABASE_URL if set. Caller decides whether to use (e.g. based on USE_TASK_QUEUE)."""
    return os.environ.get("DATABASE_URL") or None


def is_task_queue_enabled() -> bool:
    """True when USE_TASK_QUEUE is set and DATABASE_URL is present (use Redis + worker)."""
    if not get_database_url():
        return False
    return os.environ.get("USE_TASK_QUEUE", "").lower() in ("1", "true", "yes")


@contextmanager
def get_connection():
    """Context manager for a single Postgres connection. Yields None if no DATABASE_URL."""
    url = get_database_url()
    if not url:
        yield None
        return
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        yield None
        return
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def _ensure_table(conn) -> None:
    """Create tasks table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                type VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                parameters JSONB,
                result_path TEXT,
                error_message TEXT,
                rq_job_id VARCHAR(255),
                session_id VARCHAR(255),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id);
        """)
        # Progress and log columns (backward compatible: add if missing)
        for col, typ in [
            ("progress_message", "TEXT"),
            ("progress_pct", "REAL"),
            ("log_output", "TEXT"),
        ]:
            cur.execute(
                f"ALTER TABLE tasks ADD COLUMN IF NOT EXISTS {col} {typ}"
            )


def init_db() -> bool:
    """
    Create tasks table if not exists. Returns True if successful.
    Safe to call when DATABASE_URL is unset (no-op, returns False).
    """
    url = get_database_url()
    if not url:
        return False
    try:
        import psycopg2
    except ImportError:
        return False
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        try:
            _ensure_table(conn)
            return True
        finally:
            conn.close()
    except Exception:
        return False


def create_task(
    task_type: str,
    parameters: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    rq_job_id: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a task row and return its UUID string, or None if DB unavailable.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Invalid task_type: {task_type}")
    url = get_database_url()
    if not url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        return None
    task_id = str(uuid.uuid4())
    now = datetime.utcnow()
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (id, type, status, parameters, session_id, rq_job_id, created_at, updated_at)
                    VALUES (%s, %s, 'pending', %s, %s, %s, %s, %s)
                    """,
                    (task_id, task_type, Json(parameters), session_id, rq_job_id, now, now),
                )
            conn.commit()
            return task_id
        finally:
            if conn:
                conn.close()
    except Exception:
        if conn:
            conn.rollback()
        return None


def update_task_status(
    task_id: str,
    status: str,
    *,
    result_path: Optional[str] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Update task status (and optional result_path, error_message). Returns True if updated."""
    if status not in TASK_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    url = get_database_url()
    if not url:
        return False
    try:
        import psycopg2
    except ImportError:
        return False
    now = datetime.utcnow()
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                if result_path is not None and error_message is not None:
                    cur.execute(
                        "UPDATE tasks SET status = %s, result_path = %s, error_message = %s, updated_at = %s WHERE id = %s",
                        (status, result_path, error_message, now, task_id),
                    )
                elif result_path is not None:
                    cur.execute(
                        "UPDATE tasks SET status = %s, result_path = %s, updated_at = %s WHERE id = %s",
                        (status, result_path, now, task_id),
                    )
                elif error_message is not None:
                    cur.execute(
                        "UPDATE tasks SET status = %s, error_message = %s, updated_at = %s WHERE id = %s",
                        (status, error_message, now, task_id),
                    )
                else:
                    cur.execute(
                        "UPDATE tasks SET status = %s, updated_at = %s WHERE id = %s",
                        (status, now, task_id),
                    )
                n = cur.rowcount
                conn.commit()
                return n > 0
        finally:
            conn.close()
    except Exception:
        if conn:
            conn.rollback()
        return False


# Max size for log_output: keep last 50KB when appending
LOG_OUTPUT_MAX_BYTES = 50 * 1024


def update_task_progress(
    task_id: str,
    *,
    message: Optional[str] = None,
    pct: Optional[float] = None,
) -> bool:
    """Update task progress_message and/or progress_pct. Returns True if updated."""
    url = get_database_url()
    if not url:
        return False
    if message is None and pct is None:
        return True
    try:
        import psycopg2
    except ImportError:
        return False
    now = datetime.utcnow()
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                if message is not None and pct is not None:
                    cur.execute(
                        "UPDATE tasks SET progress_message = %s, progress_pct = %s, updated_at = %s WHERE id = %s",
                        (message, pct, now, task_id),
                    )
                elif message is not None:
                    cur.execute(
                        "UPDATE tasks SET progress_message = %s, updated_at = %s WHERE id = %s",
                        (message, now, task_id),
                    )
                elif pct is not None:
                    cur.execute(
                        "UPDATE tasks SET progress_pct = %s, updated_at = %s WHERE id = %s",
                        (pct, now, task_id),
                    )
                else:
                    return True
                n = cur.rowcount
                conn.commit()
                return n > 0
        finally:
            conn.close()
    except Exception:
        if conn:
            conn.rollback()
        return False


def append_task_log(task_id: str, line: str) -> bool:
    """Append a line to task log_output (with newline). Trims from start if over LOG_OUTPUT_MAX_BYTES. Returns True if updated."""
    url = get_database_url()
    if not url:
        return False
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return False
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT log_output FROM tasks WHERE id = %s",
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return False
                existing = (row["log_output"] or "") + line + "\n"
                enc = existing.encode("utf-8")
                if len(enc) > LOG_OUTPUT_MAX_BYTES:
                    # Keep last LOG_OUTPUT_MAX_BYTES; decode may drop partial char at start
                    existing = enc[-LOG_OUTPUT_MAX_BYTES:].decode("utf-8", errors="ignore")
                cur.execute(
                    "UPDATE tasks SET log_output = %s, updated_at = %s WHERE id = %s",
                    (existing, datetime.utcnow(), task_id),
                )
                conn.commit()
                return cur.rowcount > 0
        finally:
            conn.close()
    except Exception:
        if conn:
            conn.rollback()
        return False


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Return task row as dict (id, type, status, parameters, result_path, error_message, progress_message, progress_pct, log_output, created_at, updated_at)."""
    url = get_database_url()
    if not url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return None
    try:
        conn = psycopg2.connect(url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT id, type, status, parameters, result_path, error_message,
                       progress_message, progress_pct, log_output, created_at, updated_at
                       FROM tasks WHERE id = %s""",
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return dict(row)
        finally:
            conn.close()
    except Exception:
        return None


def list_recent_tasks(limit: int = 50, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent tasks (newest first). If session_id is set, only that user's tasks."""
    url = get_database_url()
    if not url:
        return []
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return []
    try:
        conn = psycopg2.connect(url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cols = "id, type, status, parameters, result_path, error_message, progress_message, progress_pct, log_output, created_at, updated_at"
                if session_id:
                    cur.execute(
                        f"""
                        SELECT {cols}
                        FROM tasks WHERE session_id = %s ORDER BY created_at DESC LIMIT %s
                        """,
                        (session_id, limit),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT {cols}
                        FROM tasks ORDER BY created_at DESC LIMIT %s
                        """,
                        (limit,),
                    )
                return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []


def delete_task(task_id: str, session_id: Optional[str] = None) -> bool:
    """Delete a task by id. When session_id is set, only delete if the task belongs to that user. Returns True if deleted."""
    url = get_database_url()
    if not url:
        return False
    try:
        import psycopg2
    except ImportError:
        return False
    conn = None
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                if session_id is not None:
                    cur.execute(
                        "DELETE FROM tasks WHERE id = %s AND session_id = %s",
                        (task_id, session_id),
                    )
                else:
                    cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
                n = cur.rowcount
                conn.commit()
                return n > 0
        finally:
            conn.close()
    except Exception:
        if conn:
            conn.rollback()
        return False
