"""
Optional app-level auth: identify the current user for per-user task visibility.
Designed to work with company auth (e.g. WebAutoAuth, OAuth2 proxy) that sets
a header with the user identity. When enabled, users see only their own tasks.
"""

import os
from typing import Optional

# Header name set by auth proxy (e.g. X-Forwarded-User, X-Auth-User). Empty = no auth filtering.
AUTH_USER_HEADER = os.environ.get("AUTH_USER_HEADER", "").strip()

# For local/dev: force a user id when no header is available (e.g. AUTH_DEFAULT_USER=dev@example.com).
AUTH_DEFAULT_USER = os.environ.get("AUTH_DEFAULT_USER", "").strip() or None


def get_current_user_id() -> Optional[str]:
    """
    Return the current user identifier, or None if auth is not configured.
    Uses (in order):
    1. HTTP header named by AUTH_USER_HEADER (when Streamlit is behind an auth proxy / WebAutoAuth).
    2. AUTH_DEFAULT_USER (for development or when proxy does not set the header).
    Streamlit 1.37+ provides st.context.headers; on older versions we fall back to AUTH_DEFAULT_USER only.
    """
    if not AUTH_USER_HEADER and not AUTH_DEFAULT_USER:
        return None
    # Try to read header (Streamlit 1.37+)
    try:
        import streamlit as st
        ctx = getattr(st, "context", None)
        headers = getattr(ctx, "headers", None) if ctx else None
        if callable(headers):
            headers = headers()
        if isinstance(headers, dict):
            value = headers.get(AUTH_USER_HEADER) or headers.get(AUTH_USER_HEADER.lower())
            if value and isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass
    return AUTH_DEFAULT_USER


def is_auth_enabled() -> bool:
    """True if AUTH_USER_HEADER or AUTH_DEFAULT_USER is set (per-user task filtering)."""
    return bool(AUTH_USER_HEADER or AUTH_DEFAULT_USER)
