from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pool_helpers import _iso_utc, _parse_utc_unix, _safe_token


@dataclass
class PoolRecord:
    request_id: str
    payload_hash: str
    request: dict[str, Any]
    priority: str
    queue_key: str
    state: str
    submitted_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    stage: str | None = None
    stage_started_at_utc: str | None = None
    retryable: bool | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    consumer_id: str = ""
    fairness_key: str = ""
    slot_affinity_requested: int | None = None
    slot_affinity_effective: int | None = None


class PoolRecordStore:
    def __init__(
        self,
        *,
        work_root: Path,
        records_max: int,
        records_ttl_s: dict[str, int],
        records_prune_interval_s: int,
    ) -> None:
        self.work_root = work_root
        self.records: dict[str, PoolRecord] = {}
        self._records_max = int(records_max)
        self._records_ttl_s = dict(records_ttl_s)
        self._records_prune_interval_s = int(records_prune_interval_s)
        self._records_pruned_total = 0
        self._records_pruned_ttl_total = 0
        self._records_pruned_overflow_total = 0
        self._records_last_prune_utc = ""
        self._records_last_prune_reason = ""
        self._records_last_prune_count = 0
        self._last_records_prune_mono = 0.0

    def get(self, request_id: str) -> PoolRecord | None:
        return self.records.get(str(request_id))

    def set(self, rec: PoolRecord) -> None:
        self.records[str(rec.request_id)] = rec

    def pop(self, request_id: str) -> PoolRecord | None:
        return self.records.pop(str(request_id), None)

    def values(self):
        return self.records.values()

    def mark_terminal(
        self,
        rec: PoolRecord,
        *,
        state: str,
        error: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        stage: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        rec.state = str(state)
        rec.finished_at_utc = _iso_utc(None)
        rec.stage = str(stage or state)
        rec.stage_started_at_utc = rec.finished_at_utc
        rec.response = dict(response or {}) if response is not None else None
        rec.error = dict(error or {}) if error is not None else None
        rec.retryable = retryable

    def to_lifecycle(self, rec: PoolRecord, *, queue_position: int | None) -> dict[str, Any]:
        return {
            "request_id": rec.request_id,
            "state": rec.state,
            "schema_version": str((rec.request or {}).get("schema_version") or ""),
            "priority": rec.priority,
            "consumer_id": str(rec.consumer_id or "unknown"),
            "fairness_key": str(rec.fairness_key or ""),
            "slot_affinity_requested": rec.slot_affinity_requested,
            "slot_affinity_effective": rec.slot_affinity_effective,
            "queue_position": queue_position,
            "submitted_at_utc": rec.submitted_at_utc,
            "started_at_utc": rec.started_at_utc,
            "finished_at_utc": rec.finished_at_utc,
            "stage": rec.stage,
            "stage_started_at_utc": rec.stage_started_at_utc,
            "retryable": rec.retryable,
            "response": rec.response,
            "error": rec.error,
        }

    def _cleanup_request_filesystem(self, request_id: str) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        roots = [
            (self.work_root / rid).resolve(),
            (self.work_root / "_uploads" / _safe_token(rid)).resolve(),
        ]
        for root in roots:
            try:
                root.relative_to(self.work_root)
            except ValueError:
                continue
            try:
                shutil.rmtree(root, ignore_errors=True)
            except Exception:
                continue

    def prune(self, *, reason: str) -> dict[str, Any] | None:
        now_unix = time.time()
        pruned_ttl = 0
        pruned_overflow = 0

        removable_by_ttl: list[str] = []
        for rid, rec in self.records.items():
            state = str(rec.state or "").strip().lower()
            if state not in {"completed", "failed", "cancelled"}:
                continue
            ttl_s = int(self._records_ttl_s.get(state, 0))
            if ttl_s <= 0:
                continue
            ref_unix = _parse_utc_unix(rec.finished_at_utc) or _parse_utc_unix(rec.submitted_at_utc)
            if ref_unix is None:
                continue
            if (now_unix - ref_unix) >= float(ttl_s):
                removable_by_ttl.append(str(rid))
        for rid in removable_by_ttl:
            self.records.pop(rid, None)
            self._cleanup_request_filesystem(rid)
            pruned_ttl += 1

        overflow = int(len(self.records) - int(self._records_max))
        if overflow > 0:
            terminal_rows: list[tuple[float, str]] = []
            for rid, rec in self.records.items():
                state = str(rec.state or "").strip().lower()
                if state not in {"completed", "failed", "cancelled"}:
                    continue
                ref_unix = _parse_utc_unix(rec.finished_at_utc) or _parse_utc_unix(rec.submitted_at_utc) or now_unix
                terminal_rows.append((float(ref_unix), str(rid)))
            terminal_rows.sort(key=lambda row: row[0])
            for _ref, rid in terminal_rows[:overflow]:
                if self.records.pop(rid, None) is not None:
                    self._cleanup_request_filesystem(rid)
                    pruned_overflow += 1

        pruned_total = int(pruned_ttl + pruned_overflow)
        if pruned_total <= 0:
            return None
        self._records_pruned_total = int(self._records_pruned_total + pruned_total)
        self._records_pruned_ttl_total = int(self._records_pruned_ttl_total + pruned_ttl)
        self._records_pruned_overflow_total = int(self._records_pruned_overflow_total + pruned_overflow)
        self._records_last_prune_utc = _iso_utc(now_unix)
        self._records_last_prune_reason = str(reason or "")
        self._records_last_prune_count = int(pruned_total)
        return {
            "reason": str(reason or ""),
            "pruned_total": int(pruned_total),
            "pruned_ttl": int(pruned_ttl),
            "pruned_overflow": int(pruned_overflow),
            "records_remaining": int(len(self.records)),
        }

    def maybe_prune(self, *, reason: str, force: bool = False) -> dict[str, Any] | None:
        now_mono = time.monotonic()
        if not force and (now_mono - float(self._last_records_prune_mono)) < float(self._records_prune_interval_s):
            return None
        self._last_records_prune_mono = float(now_mono)
        return self.prune(reason=str(reason or "periodic"))

    def stats(self) -> dict[str, Any]:
        return {
            "count": int(len(self.records)),
            "max": int(self._records_max),
            "ttl_s": {
                "completed": int(self._records_ttl_s["completed"]),
                "failed": int(self._records_ttl_s["failed"]),
                "cancelled": int(self._records_ttl_s["cancelled"]),
            },
            "prune_interval_s": int(self._records_prune_interval_s),
            "pruned_total": int(self._records_pruned_total),
            "pruned_ttl_total": int(self._records_pruned_ttl_total),
            "pruned_overflow_total": int(self._records_pruned_overflow_total),
            "last_prune_utc": str(self._records_last_prune_utc or ""),
            "last_prune_reason": str(self._records_last_prune_reason or ""),
            "last_prune_count": int(self._records_last_prune_count),
        }
