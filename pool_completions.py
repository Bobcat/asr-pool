from __future__ import annotations

from collections import deque
from typing import Any, Callable
from uuid import uuid4


class PoolCompletionFeed:
    def __init__(self, *, max_events: int, iso_utc_fn: Callable[[float | None], str]) -> None:
        self._max_events = int(max_events)
        self._iso_utc = iso_utc_fn
        self.feed_id = uuid4().hex
        self._seq_next = 1
        self._events: deque[dict[str, Any]] = deque()

    def reset(self) -> None:
        self.feed_id = uuid4().hex

    def _event_for_record(self, rec: Any) -> dict[str, Any]:
        self._seq_next = max(1, int(self._seq_next)) + 1
        seq = int(self._seq_next - 1)
        return {
            "seq": seq,
            "ts_utc": self._iso_utc(None),
            "consumer_id": str(rec.consumer_id or "unknown"),
            "request_id": str(rec.request_id),
            "state": str(rec.state),
            "priority": str(rec.priority),
            "fairness_key": str(rec.fairness_key or ""),
            "slot_affinity_requested": rec.slot_affinity_requested,
            "slot_affinity_effective": rec.slot_affinity_effective,
            "submitted_at_utc": rec.submitted_at_utc,
            "started_at_utc": rec.started_at_utc,
            "finished_at_utc": rec.finished_at_utc,
            "retryable": rec.retryable,
            "response": rec.response,
            "error": rec.error,
        }

    def append_record(self, rec: Any) -> None:
        event = self._event_for_record(rec)
        self._events.append(event)
        while len(self._events) > int(self._max_events):
            self._events.popleft()

    def collect(self, *, consumer_id: str, since_seq: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        cid = str(consumer_id or "").strip()
        safe_since = max(0, int(since_seq))
        safe_limit = max(1, min(1000, int(limit)))
        rows_window: deque[dict[str, Any]] = deque()
        for row in reversed(self._events):
            seq = max(0, int(row.get("seq") or 0))
            if seq <= safe_since:
                break
            if str(row.get("consumer_id") or "") != cid:
                continue
            rows_window.appendleft(dict(row))
            if len(rows_window) > safe_limit:
                rows_window.pop()
        rows = list(rows_window)
        next_seq = safe_since if not rows else max(0, int(rows[-1].get("seq") or safe_since))
        return rows, next_seq
