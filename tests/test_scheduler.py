from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.pool.scheduler import PoolScheduler


def _record(
    request_id: str,
    *,
    priority: str = "normal",
    state: str = "queued",
    fairness_key: str = "",
):
    queue_key = "interactive" if priority == "interactive" else "normal"
    return SimpleNamespace(
        request_id=request_id,
        priority=priority,
        queue_key=queue_key,
        state=state,
        fairness_key=fairness_key,
    )


def _scheduler(
    *,
    runner_slots: int = 2,
    interactive_reserved_slots: int = 0,
    interactive_burst_max: int = 8,
) -> PoolScheduler:
    return PoolScheduler(
        runner_slots=runner_slots,
        interactive_reserved_slots=interactive_reserved_slots,
        interactive_burst_max=interactive_burst_max,
        interactive_default_fairness_key="default-session",
    )


class SchedulerTests(unittest.TestCase):
    def test_only_interactive_and_normal_queues_are_exposed(self) -> None:
        scheduler = _scheduler()

        self.assertEqual(scheduler.queue_key_for(priority="interactive"), "interactive")
        self.assertEqual(scheduler.queue_key_for(priority="normal"), "normal")
        self.assertEqual(scheduler.queue_key_for(priority="background"), "normal")
        self.assertEqual(scheduler.queue_depth_snapshot(), {"interactive": 0, "normal": 0})

    def test_interactive_reservation_blocks_noninteractive_when_only_reserved_slot_is_free(self) -> None:
        scheduler = _scheduler(runner_slots=2, interactive_reserved_slots=1)
        records = {
            "normal-1": _record("normal-1"),
            "normal-2": _record("normal-2"),
        }
        for rec in records.values():
            scheduler.enqueue(rec)

        self.assertEqual(scheduler.dequeue_next(records=records), "normal-1")
        records["normal-1"].state = "running"

        self.assertIsNone(scheduler.dequeue_next(records=records))

    def test_interactive_can_use_reserved_slot(self) -> None:
        scheduler = _scheduler(runner_slots=2, interactive_reserved_slots=1)
        records = {
            "normal-1": _record("normal-1"),
            "normal-2": _record("normal-2"),
        }
        for rec in records.values():
            scheduler.enqueue(rec)
        self.assertEqual(scheduler.dequeue_next(records=records), "normal-1")
        records["normal-1"].state = "running"

        records["interactive-1"] = _record("interactive-1", priority="interactive", fairness_key="session-1")
        scheduler.enqueue(records["interactive-1"])

        self.assertEqual(scheduler.dequeue_next(records=records), "interactive-1")

    def test_interactive_sessions_round_robin_by_fairness_key(self) -> None:
        scheduler = _scheduler()
        records = {
            "interactive-1": _record("interactive-1", priority="interactive", fairness_key="session-1"),
            "interactive-2": _record("interactive-2", priority="interactive", fairness_key="session-1"),
            "interactive-3": _record("interactive-3", priority="interactive", fairness_key="session-2"),
        }
        for rec in records.values():
            scheduler.enqueue(rec)

        self.assertEqual(scheduler.dequeue_next(records=records), "interactive-1")
        records["interactive-1"].state = "running"
        self.assertEqual(scheduler.dequeue_next(records=records), "interactive-3")
        records["interactive-3"].state = "running"
        self.assertEqual(scheduler.dequeue_next(records=records), "interactive-2")

    def test_interactive_burst_limit_allows_noninteractive_turn(self) -> None:
        scheduler = _scheduler(runner_slots=4, interactive_reserved_slots=0, interactive_burst_max=1)
        records = {
            "interactive-1": _record("interactive-1", priority="interactive", fairness_key="session-1"),
            "interactive-2": _record("interactive-2", priority="interactive", fairness_key="session-1"),
            "normal-1": _record("normal-1"),
        }
        for rec in records.values():
            scheduler.enqueue(rec)

        self.assertEqual(scheduler.dequeue_next(records=records), "interactive-1")
        records["interactive-1"].state = "running"
        self.assertEqual(scheduler.dequeue_next(records=records), "normal-1")


if __name__ == "__main__":
    unittest.main()
