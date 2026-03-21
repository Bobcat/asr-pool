from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asr_contract import AsrRequestError, build_error_response, prepare_request
from pool_completions import PoolCompletionFeed
from pool_config import get_bool, get_float, get_int, get_str
from pool_helpers import _iso_utc, _safe_token
from pool_records import PoolRecord, PoolRecordStore
from pool_scheduler import PoolScheduler

try:
    from whisperx_runner_client import _AsrPoolWarmRunnerClient
except Exception:  # pragma: no cover - runtime import failure is handled by empty warm client pool.
    _AsrPoolWarmRunnerClient = None  # type: ignore[assignment]


INTERACTIVE_DEFAULT_FAIRNESS_KEY = "__interactive_default__"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_hash(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _seconds_between_utc(start_utc: str | None, end_utc: str | None) -> float | None:
    try:
        if not start_utc or not end_utc:
            return None
        a = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end_utc).replace("Z", "+00:00"))
        return max(0.0, float((b - a).total_seconds()))
    except Exception:
        return None


class AsrPoolService:
    def __init__(self) -> None:
        self._runner_slots = get_int("scheduler.runner_slots", 2, min_value=1)
        self._queue_limits = {
            "interactive": get_int("scheduler.queue_limits.interactive", 8, min_value=1),
            "normal": get_int("scheduler.queue_limits.normal", 20, min_value=1),
            "background": get_int("scheduler.queue_limits.background", 50, min_value=1),
        }
        self._timeouts_s = {
            "interactive": get_int("scheduler.request_timeouts_s.interactive", 30, min_value=1),
            "normal": get_int("scheduler.request_timeouts_s.normal", 120, min_value=1),
            "background": get_int("scheduler.request_timeouts_s.background", 300, min_value=1),
        }
        self._warm_start_enabled = get_bool("lifecycle.warm_start.enabled", True)
        self._warm_start_timeout_s = get_int("lifecycle.warm_start.timeout_s", 180, min_value=1)
        self._watchdog_enabled = get_bool("lifecycle.watchdog.enabled", True)
        self._watchdog_interval_s = max(
            0.2,
            get_float("lifecycle.watchdog.poll_ms", 2000, min_value=200) / 1000.0,
        )
        self._watchdog_recover_timeout_s = get_int("lifecycle.watchdog.recover_timeout_s", 30, min_value=1)
        records_max = get_int("records.max", 10000, min_value=100)
        records_ttl_s = {
            "completed": get_int("records.ttl_s.completed", 900, min_value=10),
            "failed": get_int("records.ttl_s.failed", 1800, min_value=10),
            "cancelled": get_int("records.ttl_s.cancelled", 600, min_value=10),
        }
        records_prune_interval_s = get_int("records.prune_interval_s", 30, min_value=1)
        work_root_cfg = get_str("paths.work_root", "")
        if work_root_cfg:
            self._work_root = Path(work_root_cfg).expanduser().resolve()
        else:
            self._work_root = (_repo_root() / "data" / "asr_pool").resolve()
        self._work_root.mkdir(parents=True, exist_ok=True)
        self._interactive_burst_max = get_int("scheduler.interactive_burst_max", 8, min_value=1)
        self._completion_events_max = get_int("completions.max_events", 20000, min_value=1000)
        self._warm_clients: list[Any] = []
        if _AsrPoolWarmRunnerClient is not None:
            try:
                self._warm_clients = [_AsrPoolWarmRunnerClient() for _ in range(self._runner_slots)]
            except Exception:
                self._warm_clients = []
        self._scheduler = PoolScheduler(
            interactive_burst_max=self._interactive_burst_max,
            interactive_default_fairness_key=INTERACTIVE_DEFAULT_FAIRNESS_KEY,
        )
        self._completion_feed = PoolCompletionFeed(max_events=self._completion_events_max, iso_utc_fn=_iso_utc)
        self._record_store = PoolRecordStore(
            work_root=self._work_root,
            records_max=records_max,
            records_ttl_s=records_ttl_s,
            records_prune_interval_s=records_prune_interval_s,
        )
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False
        self._watchdog_restart_count: list[int] = [0 for _ in range(max(0, int(self._runner_slots)))]
        self._stage_poll_interval_s = max(
            0.05,
            get_float("scheduler.stage_poll_ms", 150, min_value=50) / 1000.0,
        )

    async def start(self) -> None:
        should_prewarm = False
        async with self._lock:
            if self._tasks:
                return
            self._completion_feed.reset()
            self._stopping = False
            for idx in range(self._runner_slots):
                task = asyncio.create_task(self._runner_loop(idx), name=f"asr-pool-runner-{idx}")
                self._tasks.append(task)
            should_prewarm = bool(self._warm_clients) and bool(self._warm_start_enabled)
            self._emit_event(
                "pool_started",
                runner_slots=int(self._runner_slots),
                executor_mode="warm_local",
                warm_start_enabled=bool(self._warm_start_enabled),
                watchdog_enabled=bool(self._watchdog_enabled),
                watchdog_interval_s=round(float(self._watchdog_interval_s), 3),
                watchdog_recover_timeout_s=int(self._watchdog_recover_timeout_s),
                records_max=int(self._record_store.stats()["max"]),
                records_ttl_completed_s=int(self._record_store.stats()["ttl_s"]["completed"]),
                records_ttl_failed_s=int(self._record_store.stats()["ttl_s"]["failed"]),
                records_ttl_cancelled_s=int(self._record_store.stats()["ttl_s"]["cancelled"]),
                interactive_burst_max=int(self._interactive_burst_max),
            )
            if bool(self._watchdog_enabled):
                watchdog_task = asyncio.create_task(self._watchdog_loop(), name="asr-pool-watchdog")
                self._tasks.append(watchdog_task)
        if should_prewarm:
            await self._prewarm_runners()

    async def stop(self) -> None:
        async with self._lock:
            self._stopping = True
            self._cond.notify_all()
            tasks = list(self._tasks)
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for client in list(self._warm_clients):
            try:
                client.shutdown(reason="pool_shutdown")
            except Exception:
                continue
        self._emit_event("pool_stopped")

    async def _prewarm_runners(self) -> None:
        async def _run_one(slot_idx: int, client: Any) -> None:
            try:
                await asyncio.to_thread(self._prewarm_one_runner, slot_idx, client)
            except Exception as e:
                try:
                    print(f"asr_pool prewarm slot={slot_idx} failed: {type(e).__name__}: {e}", flush=True)
                except Exception:
                    pass

        coros = [_run_one(idx, client) for idx, client in enumerate(list(self._warm_clients))]
        if not coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=float(self._warm_start_timeout_s),
            )
            self._emit_event(
                "pool_prewarm_done",
                slots=int(len(coros)),
                timeout_s=int(self._warm_start_timeout_s),
            )
        except asyncio.TimeoutError:
            try:
                print(
                    f"asr_pool prewarm timeout after {self._warm_start_timeout_s}s (slots={len(coros)})",
                    flush=True,
                )
            except Exception:
                pass
            self._emit_event(
                "pool_prewarm_timeout",
                slots=int(len(coros)),
                timeout_s=int(self._warm_start_timeout_s),
            )

    @staticmethod
    def _warm_client_health(client: Any) -> tuple[bool, int | None]:
        with client._lock:
            proc = client._proc
            if proc is None:
                return False, None
            pid = int(getattr(proc, "pid", 0) or 0) or None
            try:
                alive = bool(proc.poll() is None)
            except Exception:
                alive = False
            return alive, pid

    @staticmethod
    def _recover_warm_client(slot_idx: int, client: Any, *, reason: str) -> dict[str, Any]:
        with client._lock:
            old_proc = client._proc
            old_pid = int(getattr(old_proc, "pid", 0) or 0) or None
            if old_proc is not None:
                try:
                    if old_proc.poll() is not None:
                        client._shutdown_locked(reason=f"watchdog_{reason}")
                except Exception:
                    pass
        client.prewarm()
        alive, new_pid = AsrPoolService._warm_client_health(client)
        if not alive:
            raise RuntimeError("watchdog_recovery_runner_not_alive")
        return {
            "slot_idx": int(slot_idx),
            "old_pid": old_pid,
            "new_pid": int(new_pid or 0) or None,
        }

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(float(self._watchdog_interval_s))
            if self._stopping:
                return
            if not self._warm_clients:
                continue
            for slot_idx, client in enumerate(list(self._warm_clients)):
                if self._stopping:
                    return
                alive, pid = await asyncio.to_thread(self._warm_client_health, client)
                if alive:
                    continue
                self._emit_event(
                    "runner_watchdog_detected_unhealthy",
                    slot_idx=int(slot_idx),
                    pid=(int(pid) if pid is not None else None),
                )
                try:
                    rec = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._recover_warm_client,
                            int(slot_idx),
                            client,
                            reason="unhealthy",
                        ),
                        timeout=float(self._watchdog_recover_timeout_s),
                    )
                    self._watchdog_restart_count[slot_idx] = int(self._watchdog_restart_count[slot_idx]) + 1
                    self._emit_event(
                        "runner_watchdog_recovered",
                        slot_idx=int(slot_idx),
                        old_pid=rec.get("old_pid"),
                        new_pid=rec.get("new_pid"),
                        restart_count=int(self._watchdog_restart_count[slot_idx]),
                    )
                except asyncio.TimeoutError:
                    self._emit_event(
                        "runner_watchdog_recover_timeout",
                        slot_idx=int(slot_idx),
                        timeout_s=int(self._watchdog_recover_timeout_s),
                    )
                except Exception as e:
                    self._emit_event(
                        "runner_watchdog_recover_failed",
                        slot_idx=int(slot_idx),
                        error=f"{type(e).__name__}: {e}",
                    )

    def _prewarm_one_runner(self, slot_idx: int, client: Any) -> None:
        try:
            client.prewarm()
        except Exception as e:
            raise RuntimeError(f"slot={slot_idx}: {type(e).__name__}: {e}") from e

    async def submit(self, raw_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            prepared = prepare_request(raw_payload)
        except AsrRequestError as e:
            self._emit_event(
                "submit_rejected_validation",
                code=str(e.code),
                message=str(e),
            )
            return 400, {
                "code": str(e.code),
                "message": str(e),
                "retryable": False,
                "details": dict(e.details or {}),
            }

        request_id = prepared["request_id"]
        payload_hash = _json_hash(prepared)
        priority = prepared["priority"]
        queue_key = self._scheduler.queue_key_for(priority=priority)
        consumer_id = self._consumer_id_from_request(prepared)
        fairness_key, slot_affinity_requested, slot_affinity_effective = self._extract_routing_metadata(prepared)
        if not fairness_key:
            fairness_key = INTERACTIVE_DEFAULT_FAIRNESS_KEY

        async with self._lock:
            self._maybe_prune_records_unlocked(reason="submit", force=False)
            existing = self._record_store.get(request_id)
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    self._emit_event(
                        "submit_conflict",
                        request_id=request_id,
                        priority=priority,
                    )
                    return 409, {
                        "code": "ASR_REQUEST_ID_CONFLICT",
                        "message": "request_id already exists with different payload",
                        "retryable": False,
                        "details": {"request_id": request_id},
                    }
                self._emit_event(
                    "submit_idempotent_hit",
                    request_id=request_id,
                    state=str(existing.state),
                )
                return 200, self._to_lifecycle(existing)

            if self._priority_depth(priority) >= int(self._queue_limits.get(priority, 1)):
                self._emit_event(
                    "submit_rejected_queue_full",
                    request_id=request_id,
                    priority=priority,
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                    queue_limit=int(self._queue_limits.get(priority, 1)),
                )
                return 429, {
                    "code": "ASR_QUEUE_FULL",
                    "message": f"{priority} queue depth limit reached",
                    "retryable": True,
                    "details": {
                        "priority": priority,
                        "queue_depth": int(self._priority_depth(priority)),
                        "queue_limit": int(self._queue_limits.get(priority, 1)),
                    },
                }

            rec = PoolRecord(
                request_id=request_id,
                payload_hash=payload_hash,
                request=prepared,
                priority=priority,
                queue_key=queue_key,
                state="queued",
                submitted_at_utc=_iso_utc(),
                consumer_id=consumer_id,
                fairness_key=str(fairness_key or ""),
                slot_affinity_requested=(None if slot_affinity_requested is None else int(slot_affinity_requested)),
                slot_affinity_effective=(None if slot_affinity_effective is None else int(slot_affinity_effective)),
            )
            self._record_store.set(rec)
            self._scheduler.enqueue(rec)
            queue_position = self._scheduler.queue_position(rec)
            self._emit_event(
                "submit_accepted",
                request_id=request_id,
                priority=rec.priority,
                consumer_id=str(rec.consumer_id or "unknown"),
                fairness_key=str(rec.fairness_key or ""),
                slot_affinity_requested=rec.slot_affinity_requested,
                slot_affinity_effective=rec.slot_affinity_effective,
                queue_key=queue_key,
                queue_position=queue_position,
                queue_depth=self._queue_depth_snapshot_unlocked(),
            )
            self._cond.notify_all()
            return 202, self._to_lifecycle(rec)

    async def get_request(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            self._maybe_prune_records_unlocked(reason="get_request", force=False)
            rec = self._record_store.get(rid)
            if rec is None:
                return 404, {
                    "code": "ASR_REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            return 200, self._to_lifecycle(rec)

    async def cancel(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            self._maybe_prune_records_unlocked(reason="cancel", force=False)
            rec = self._record_store.get(rid)
            if rec is None:
                self._emit_event("cancel_not_found", request_id=rid)
                return 404, {
                    "code": "ASR_REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            if rec.state == "queued":
                self._scheduler.remove(rid, rec.queue_key)
                self._mark_record_terminal_unlocked(
                    rec,
                    state="cancelled",
                    stage="cancelled",
                    retryable=None,
                )
                self._emit_event(
                    "cancel_queued",
                    request_id=rec.request_id,
                    priority=rec.priority,
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                )
            elif rec.state == "running":
                rec.state = "cancel_requested"
                self._emit_event(
                    "cancel_running",
                    request_id=rec.request_id,
                    priority=rec.priority,
                )
            else:
                self._emit_event(
                    "cancel_noop",
                    request_id=rec.request_id,
                    state=str(rec.state),
                )
            return 200, {
                "request_id": rec.request_id,
                "state": rec.state,
                "message": "cancel accepted",
            }

    async def completions_wait(
        self,
        *,
        consumer_id: str,
        since_seq: int,
        limit: int,
        wait_timeout_s: float = 0.0,
    ) -> tuple[int, dict[str, Any]]:
        cid = str(consumer_id or "").strip()
        if not cid:
            return 400, {
                "code": "ASR_COMPLETIONS_CONSUMER_REQUIRED",
                "message": "consumer_id is required",
                "retryable": False,
                "details": {},
            }
        safe_since = max(0, int(since_seq))
        safe_limit = max(1, min(1000, int(limit)))
        safe_wait_timeout_s = max(0.0, float(wait_timeout_s))
        wait_deadline = (time.monotonic() + safe_wait_timeout_s) if safe_wait_timeout_s > 0.0 else 0.0
        async with self._cond:
            while True:
                self._maybe_prune_records_unlocked(reason="completions", force=False)
                rows, next_seq = self._completion_feed.collect(
                    consumer_id=cid,
                    since_seq=safe_since,
                    limit=safe_limit,
                )
                if rows or safe_wait_timeout_s <= 0.0:
                    return 200, {
                        "feed_id": str(self._completion_feed.feed_id),
                        "consumer_id": cid,
                        "since_seq": int(safe_since),
                        "next_seq": int(next_seq),
                        "events": rows,
                    }
                remaining_s = float(wait_deadline - time.monotonic())
                if remaining_s <= 0.0:
                    return 200, {
                        "feed_id": str(self._completion_feed.feed_id),
                        "consumer_id": cid,
                        "since_seq": int(safe_since),
                        "next_seq": int(next_seq),
                        "events": [],
                    }
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining_s)
                except asyncio.TimeoutError:
                    return 200, {
                        "feed_id": str(self._completion_feed.feed_id),
                        "consumer_id": cid,
                        "since_seq": int(safe_since),
                        "next_seq": int(next_seq),
                        "events": [],
                    }

    async def completions(self, *, consumer_id: str, since_seq: int, limit: int) -> tuple[int, dict[str, Any]]:
        return await self.completions_wait(
            consumer_id=str(consumer_id),
            since_seq=int(since_seq),
            limit=int(limit),
            wait_timeout_s=0.0,
        )

    async def pending_status(
        self,
        *,
        consumer_id: str,
        request_ids: list[str],
        limit: int,
    ) -> tuple[int, dict[str, Any]]:
        cid = str(consumer_id or "").strip()
        if not cid:
            return 400, {
                "code": "ASR_PENDING_STATUS_CONSUMER_REQUIRED",
                "message": "consumer_id is required",
                "retryable": False,
                "details": {},
            }
        safe_limit = max(1, min(1000, int(limit)))
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for raw in list(request_ids or []):
            rid = str(raw or "").strip()
            if not rid:
                continue
            if rid in seen:
                continue
            seen.add(rid)
            normalized_ids.append(rid)
            if len(normalized_ids) >= safe_limit:
                break
        async with self._lock:
            self._maybe_prune_records_unlocked(reason="pending_status", force=False)
            rows: list[dict[str, Any]] = []
            for rid in normalized_ids:
                rec = self._record_store.get(rid)
                if rec is None:
                    continue
                if str(rec.consumer_id or "") != cid:
                    continue
                rows.append(
                    {
                        "request_id": rec.request_id,
                        "state": str(rec.state),
                        "stage": str(rec.stage or ""),
                        "stage_started_at_utc": rec.stage_started_at_utc,
                        "submitted_at_utc": rec.submitted_at_utc,
                        "started_at_utc": rec.started_at_utc,
                        "finished_at_utc": rec.finished_at_utc,
                        "queue_position": self._to_lifecycle(rec).get("queue_position"),
                    }
                )
            return 200, {
                "feed_id": str(self._completion_feed.feed_id),
                "consumer_id": cid,
                "request_count": int(len(normalized_ids)),
                "rows": rows,
            }

    async def pool_status(self) -> dict[str, Any]:
        async with self._lock:
            queued_interactive = self._priority_depth("interactive")
            queued_normal = self._priority_depth("normal")
            queued_background = self._priority_depth("background")
            running = sum(1 for rec in self._record_store.values() if rec.state in {"running", "cancel_requested"})
            return {
                "service": "asr-runtime-pool",
                "version": "1.0.0",
                "now_utc": _iso_utc(),
                "slots_total": int(self._runner_slots),
                "slots_busy": int(running),
                "slots_available": int(max(0, self._runner_slots - running)),
                "slots_by_priority": {
                    "interactive": int(running),
                    "normal": 0,
                    "background": 0,
                },
                "queue_limits": dict(self._queue_limits),
                "queue_depth": {
                    "interactive": int(queued_interactive),
                    "normal": int(queued_normal),
                    "background": int(queued_background),
                },
                "request_timeouts_s": dict(self._timeouts_s),
                "queue_wait_ms_p95": {},
                "blob_fetch_ms_p95": None,
                "watchdog": {
                    "enabled": bool(self._watchdog_enabled),
                    "interval_s": round(float(self._watchdog_interval_s), 3),
                    "recover_timeout_s": int(self._watchdog_recover_timeout_s),
                    "restarts_by_slot": [int(v) for v in self._watchdog_restart_count],
                },
                "records": self._record_store.stats(),
                "scheduling_policy": {
                    "interactive_single_queue": True,
                    "interactive_round_robin_by_session": True,
                    "interactive_default_fairness_key": str(INTERACTIVE_DEFAULT_FAIRNESS_KEY),
                    "interactive_burst_max": int(self._interactive_burst_max),
                    "fairness_mode": "burst_then_round_robin_interactive_sessions_and_noninteractive_priorities",
                },
            }

    def _priority_depth(self, priority: str) -> int:
        return int(self._scheduler.priority_depth(priority))

    def _queue_depth_snapshot_unlocked(self) -> dict[str, int]:
        return self._scheduler.queue_depth_snapshot()

    def _has_running_background_unlocked(self) -> bool:
        return self._scheduler.has_running_background(self._record_store.records)

    def _emit_event(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts_utc": _iso_utc(),
            "component": "asr_runtime_pool",
            "event": str(event),
        }
        payload.update({k: v for k, v in fields.items() if v is not None})
        try:
            print("ASR_POOL_EVENT " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception:
            pass

    @staticmethod
    def _consumer_id_from_request(request: dict[str, Any]) -> str:
        raw = str(request.get("consumer_id") or "").strip()
        return raw or "unknown"

    @staticmethod
    def _extract_routing_metadata(request: dict[str, Any]) -> tuple[str, int | None, int | None]:
        routing = dict(request.get("routing") or {})
        fairness_key = str(routing.get("fairness_key") or "").strip()
        slot_affinity_requested: int | None = None
        if "slot_affinity" in routing and routing.get("slot_affinity") is not None:
            try:
                slot_affinity_requested = int(routing.get("slot_affinity"))
            except Exception:
                slot_affinity_requested = None
        slot_affinity_effective: int | None = 0 if slot_affinity_requested == 0 else None
        return fairness_key, slot_affinity_requested, slot_affinity_effective

    def _append_completion_event_unlocked(self, rec: PoolRecord) -> None:
        self._completion_feed.append_record(rec)
        self._cond.notify_all()

    def _mark_record_terminal_unlocked(
        self,
        rec: PoolRecord,
        *,
        state: str,
        error: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        stage: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        self._record_store.mark_terminal(
            rec,
            state=state,
            error=error,
            response=response,
            stage=stage,
            retryable=retryable,
        )
        self._append_completion_event_unlocked(rec)

    def _to_lifecycle(self, rec: PoolRecord) -> dict[str, Any]:
        return self._record_store.to_lifecycle(rec, queue_position=self._scheduler.queue_position(rec))

    def _emit_prune_event(self, prune_info: dict[str, Any] | None) -> int:
        if not prune_info:
            return 0
        self._emit_event("records_pruned", **prune_info)
        return int(prune_info.get("pruned_total") or 0)

    def _maybe_prune_records_unlocked(self, *, reason: str, force: bool = False) -> int:
        prune_info = self._record_store.maybe_prune(reason=reason, force=force)
        return self._emit_prune_event(prune_info)

    async def _poll_stage_updates(self, *, request_id: str, progress_path: Path, stop_event: asyncio.Event) -> None:
        last_stage = ""
        while True:
            if stop_event.is_set():
                break
            try:
                obj = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
                stage = str(obj.get("stage") or "").strip().lower()
                stage_ts = str(obj.get("ts_utc") or "").strip()
                if stage and stage != last_stage:
                    async with self._lock:
                        rec = self._record_store.get(str(request_id))
                        if rec is not None and rec.state in {"running", "cancel_requested"}:
                            rec.stage = str(stage)
                            rec.stage_started_at_utc = str(stage_ts or _iso_utc())
                    self._emit_event(
                        "request_stage",
                        request_id=str(request_id),
                        stage=str(stage),
                    )
                    last_stage = str(stage)
            except Exception:
                pass
            await asyncio.sleep(float(self._stage_poll_interval_s))

    async def _dequeue_next_request_id(self, slot_idx: int) -> str:
        async with self._cond:
            while True:
                if self._stopping:
                    raise asyncio.CancelledError()
                rid = self._scheduler.dequeue_next(slot_idx=slot_idx, records=self._record_store.records)
                if rid:
                    return rid
                await self._cond.wait()

    async def _runner_loop(self, slot_idx: int) -> None:
        while True:
            rid = await self._dequeue_next_request_id(slot_idx)
            progress_path = (self._work_root / str(rid) / f"slot_{slot_idx}" / "_progress.json").resolve()
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.unlink(missing_ok=True)
            except Exception:
                pass
            async with self._lock:
                rec = self._record_store.get(rid)
                if rec is None:
                    continue
                if rec.state != "queued":
                    continue
                rec.state = "running"
                rec.started_at_utc = _iso_utc()
                rec.stage = "dispatch"
                rec.stage_started_at_utc = rec.started_at_utc
                request = dict(rec.request)
                timeout_s = int(self._timeouts_s.get(rec.priority, 120))
                queue_wait_s = _seconds_between_utc(rec.submitted_at_utc, rec.started_at_utc)
                self._emit_event(
                    "request_started",
                    request_id=rec.request_id,
                    priority=rec.priority,
                    queue_key=rec.queue_key,
                    slot_idx=int(slot_idx),
                    timeout_s=int(timeout_s),
                    queue_wait_s=(round(float(queue_wait_s), 3) if queue_wait_s is not None else None),
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                )

            stage_stop = asyncio.Event()
            stage_task = asyncio.create_task(
                self._poll_stage_updates(
                    request_id=str(rid),
                    progress_path=progress_path,
                    stop_event=stage_stop,
                ),
                name=f"asr-pool-stage-{slot_idx}-{rid}",
            )
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._execute_request,
                        request=request,
                        slot_idx=slot_idx,
                        progress_path=progress_path,
                    ),
                    timeout=float(timeout_s),
                )
            except asyncio.TimeoutError:
                response = build_error_response(
                    request=request,
                    code="ASR_REQUEST_TIMEOUT",
                    message=f"ASR request exceeded timeout ({timeout_s}s)",
                    retryable=True,
                    details={"timeout_s": int(timeout_s)},
                )
            except Exception as e:
                response = build_error_response(
                    request=request,
                    code="ASR_RUNTIME_FAILURE",
                    message=f"ASR runtime failure: {type(e).__name__}: {e}",
                    retryable=True,
                    details={"exc_type": type(e).__name__},
                )
            finally:
                stage_stop.set()
                try:
                    await stage_task
                except Exception:
                    pass

            ok = bool(response.get("ok", False))
            async with self._lock:
                rec2 = self._record_store.get(rid)
                if rec2 is None:
                    continue
                if rec2.state == "cancel_requested":
                    self._mark_record_terminal_unlocked(
                        rec2,
                        state="cancelled",
                        stage="cancelled",
                        retryable=None,
                    )
                else:
                    err_obj = dict(response.get("error") or {}) if not ok else None
                    retryable = bool((err_obj or {}).get("retryable", False)) if not ok else None
                    self._mark_record_terminal_unlocked(
                        rec2,
                        state=("completed" if ok else "failed"),
                        stage=("completed" if ok else "failed"),
                        response=dict(response),
                        error=err_obj,
                        retryable=retryable,
                    )
                runtime_meta = dict((rec2.response or {}).get("runtime") or {})
                err = dict(rec2.error or {})
                exec_s = _seconds_between_utc(rec2.started_at_utc, rec2.finished_at_utc)
                self._emit_event(
                    "request_finished",
                    request_id=rec2.request_id,
                    priority=rec2.priority,
                    slot_idx=int(slot_idx),
                    state=str(rec2.state),
                    ok=bool(ok),
                    retryable=(None if rec2.retryable is None else bool(rec2.retryable)),
                    execution_s=(round(float(exec_s), 3) if exec_s is not None else None),
                    error_code=(str(err.get("code")) if err else None),
                    error_message=(str(err.get("message")) if err else None),
                    runner_kind=(str(runtime_meta.get("runner_kind")) if runtime_meta else None),
                    runner_reused=(runtime_meta.get("runner_reused") if runtime_meta else None),
                    transport=(str(runtime_meta.get("transport")) if runtime_meta else None),
                )
                self._maybe_prune_records_unlocked(reason="request_finished", force=False)

    def _execute_request(self, *, request: dict[str, Any], slot_idx: int, progress_path: Path | None = None) -> dict[str, Any]:
        request = dict(request or {})
        request_id = str(request.get("request_id") or f"req_{int(time.time())}")
        job_root = (self._work_root / request_id / f"slot_{slot_idx}").resolve()
        out_dir = (job_root / "whisperx").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        job = SimpleNamespace(
            whisperx_dir=out_dir,
        )
        if slot_idx < 0 or slot_idx >= len(self._warm_clients):
            return build_error_response(
                request=request,
                code="ASR_POOL_RUNNER_INVALID_SLOT",
                message=f"Warm runner slot out of range: {slot_idx}",
                retryable=True,
                details={"slot_idx": int(slot_idx), "slots": int(len(self._warm_clients))},
            )
        client = self._warm_clients[slot_idx]
        try:
            return dict(client.transcribe(job=job, request=request, progress_path=progress_path) or {})
        except Exception as e:
            return build_error_response(
                request=request,
                code="ASR_POOL_WARM_EXECUTOR_FAILURE",
                message=f"Warm executor failed: {type(e).__name__}: {e}",
                retryable=True,
                details={"slot_idx": int(slot_idx), "exc_type": type(e).__name__},
            )
