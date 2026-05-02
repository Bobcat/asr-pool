from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.pool.completions import PoolCompletionFeed
from app.pool.records import PoolRecord, PoolRecordStore


def _record(*, state: str = "queued") -> PoolRecord:
    return PoolRecord(
        request_id="req-1",
        payload_hash="hash-1",
        request={"schema_version": "asr_v2"},
        priority="normal",
        queue_key="normal",
        state=state,
        submitted_at_utc="2026-05-02T12:00:00Z",
        consumer_id="consumer-1",
        fairness_key="session-1",
    )


class RecordPayloadTests(unittest.TestCase):
    def test_lifecycle_payload_excludes_obsolete_slot_affinity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PoolRecordStore(
                work_root=Path(tmp),
                records_max=100,
                records_ttl_s={"completed": 10, "failed": 10, "cancelled": 10},
                records_prune_interval_s=1,
            )

            payload = store.to_lifecycle(_record(), queue_position=3)

        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["priority"], "normal")
        self.assertEqual(payload["fairness_key"], "session-1")
        self.assertEqual(payload["queue_position"], 3)
        self.assertNotIn("slot_affinity_requested", payload)
        self.assertNotIn("slot_affinity_effective", payload)

    def test_completion_payload_excludes_obsolete_slot_affinity_fields(self) -> None:
        feed = PoolCompletionFeed(max_events=10, iso_utc_fn=lambda _ts: "2026-05-02T12:00:01Z")
        feed.append_record(_record(state="completed"))

        rows, next_seq = feed.collect(consumer_id="consumer-1", since_seq=0, limit=10)

        self.assertEqual(next_seq, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "req-1")
        self.assertEqual(rows[0]["priority"], "normal")
        self.assertEqual(rows[0]["fairness_key"], "session-1")
        self.assertNotIn("slot_affinity_requested", rows[0])
        self.assertNotIn("slot_affinity_effective", rows[0])


if __name__ == "__main__":
    unittest.main()
