from __future__ import annotations

import re
import time
from datetime import datetime, timezone


_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _iso_utc(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_unix(value: str | None) -> float | None:
    try:
        if not value:
            return None
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return float(dt.timestamp())
    except Exception:
        return None


def _safe_token(value: str, *, fallback: str = "request") -> str:
    text = _SAFE_TOKEN_RE.sub("_", str(value or "").strip())
    text = text.strip("._-")
    return text or fallback
