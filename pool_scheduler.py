from __future__ import annotations

from collections import deque
from typing import Any


class PoolScheduler:
    def __init__(self, *, interactive_burst_max: int, interactive_default_fairness_key: str) -> None:
        self._interactive_default_fairness_key = str(interactive_default_fairness_key)
        self._interactive_burst_max = int(interactive_burst_max)
        self.queues: dict[str, deque[str]] = {
            "interactive": deque(),
            "normal": deque(),
            "background": deque(),
        }
        self._interactive_burst_count = 0
        self._interactive_rr_last_session_key = ""
        self._noninteractive_next = "normal"

    def queue_key_for(self, *, priority: str) -> str:
        p = str(priority or "normal").strip().lower() or "normal"
        if p == "interactive":
            return "interactive"
        if p == "background":
            return "background"
        return "normal"

    def enqueue(self, rec: Any) -> None:
        self.queues[str(rec.queue_key)].append(str(rec.request_id))

    def priority_depth(self, priority: str) -> int:
        p = str(priority or "").strip().lower()
        if p == "interactive":
            return int(len(self.queues["interactive"]))
        if p == "background":
            return int(len(self.queues["background"]))
        return int(len(self.queues["normal"]))

    def queue_depth_snapshot(self) -> dict[str, int]:
        return {
            "interactive": int(self.priority_depth("interactive")),
            "normal": int(self.priority_depth("normal")),
            "background": int(self.priority_depth("background")),
        }

    def has_running_background(self, records: dict[str, Any]) -> bool:
        for rec in records.values():
            if str(rec.state) == "running" and str(rec.priority) == "background":
                return True
        return False

    def queue_position(self, rec: Any) -> int | None:
        if str(rec.state) != "queued":
            return None
        queue = self.queues.get(str(rec.queue_key))
        if queue is None:
            return None
        try:
            return int(list(queue).index(str(rec.request_id)) + 1)
        except ValueError:
            return None

    def remove(self, request_id: str, queue_key: str) -> None:
        queue = self.queues.get(str(queue_key))
        if queue is None:
            return
        try:
            queue.remove(str(request_id))
        except ValueError:
            return
        if str(queue_key) == "interactive" and not queue:
            self._interactive_rr_last_session_key = ""

    def _interactive_session_key_for_record(self, rec: Any) -> str:
        fairness_key = str(rec.fairness_key or "").strip()
        if fairness_key:
            return fairness_key
        return self._interactive_default_fairness_key

    def _interactive_sessions_snapshot(self, records: dict[str, Any]) -> list[str]:
        queue = self.queues["interactive"]
        stale_ids: list[str] = []
        seen: set[str] = set()
        sessions: list[str] = []
        for rid in list(queue):
            rec = records.get(str(rid))
            if rec is None or rec.state != "queued" or rec.queue_key != "interactive":
                stale_ids.append(str(rid))
                continue
            session_key = self._interactive_session_key_for_record(rec)
            if session_key in seen:
                continue
            seen.add(session_key)
            sessions.append(session_key)
        for rid in stale_ids:
            try:
                queue.remove(rid)
            except ValueError:
                continue
        if not queue:
            self._interactive_rr_last_session_key = ""
        return sessions

    def _dequeue_interactive_request_id(self, records: dict[str, Any]) -> str | None:
        queue = self.queues["interactive"]
        if not queue:
            self._interactive_rr_last_session_key = ""
            return None
        sessions = self._interactive_sessions_snapshot(records)
        if not sessions:
            return None

        preferred_session = sessions[0]
        last_session = str(self._interactive_rr_last_session_key or "")
        if last_session and len(sessions) > 1 and last_session in sessions:
            idx = int(sessions.index(last_session))
            preferred_session = sessions[(idx + 1) % len(sessions)]
        elif last_session and last_session not in sessions:
            self._interactive_rr_last_session_key = ""

        ordered_sessions = [preferred_session] + [s for s in sessions if s != preferred_session]
        for session_key in ordered_sessions:
            for rid in list(queue):
                rec = records.get(str(rid))
                if rec is None or rec.state != "queued" or rec.queue_key != "interactive":
                    continue
                if self._interactive_session_key_for_record(rec) != session_key:
                    continue
                try:
                    queue.remove(str(rid))
                except ValueError:
                    continue
                self._interactive_rr_last_session_key = str(session_key)
                return str(rid)
        return None

    def _noninteractive_order(self) -> list[str]:
        if str(self._noninteractive_next) == "background":
            return ["background", "normal"]
        return ["normal", "background"]

    def _dequeue_order(self) -> list[str]:
        interactive_ready = self.priority_depth("interactive") > 0
        normal_ready = self.priority_depth("normal") > 0
        background_ready = self.priority_depth("background") > 0
        noninteractive_ready = normal_ready or background_ready
        prefer_noninteractive = (
            interactive_ready
            and noninteractive_ready
            and int(self._interactive_burst_count) >= int(self._interactive_burst_max)
        )
        if prefer_noninteractive:
            return self._noninteractive_order() + ["interactive"]
        return ["interactive"] + self._noninteractive_order()

    def _note_dequeue_key(self, queue_key: str) -> None:
        key = str(queue_key or "").strip().lower()
        if key == "interactive" or key.startswith("interactive_"):
            self._interactive_burst_count = int(self._interactive_burst_count) + 1
            return
        self._interactive_burst_count = 0
        if key == "normal":
            self._noninteractive_next = "background"
        elif key == "background":
            self._noninteractive_next = "normal"

    def dequeue_next(self, *, slot_idx: int, records: dict[str, Any]) -> str | None:
        for key in self._dequeue_order():
            if key == "background" and self.has_running_background(records):
                continue
            if key == "interactive":
                rid = self._dequeue_interactive_request_id(records)
                if rid:
                    self._note_dequeue_key(key)
                    return rid
                continue
            queue = self.queues[key]
            while queue:
                rid = queue.popleft()
                rec = records.get(rid)
                if rec is None:
                    continue
                if rec.state != "queued":
                    continue
                if rec.slot_affinity_effective is not None and int(slot_idx) != int(rec.slot_affinity_effective):
                    queue.appendleft(rid)
                    break
                if key == "background" and self.has_running_background(records):
                    queue.appendleft(rid)
                    break
                self._note_dequeue_key(key)
                return rid
        return None
