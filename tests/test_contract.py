from __future__ import annotations

import unittest

from app.asr.contract import AsrRequestError, prepare_request


def _request(**overrides):
    payload = {
        "schema_version": "asr_v2",
        "request_id": "req-1",
        "audio": {"local_path": "/tmp/input.wav"},
    }
    payload.update(overrides)
    return payload


class ContractTests(unittest.TestCase):
    def test_priority_defaults_to_normal(self) -> None:
        prepared = prepare_request(_request())

        self.assertEqual(prepared["priority"], "normal")

    def test_interactive_priority_is_preserved(self) -> None:
        prepared = prepare_request(_request(priority="interactive"))

        self.assertEqual(prepared["priority"], "interactive")

    def test_noninteractive_priorities_normalize_to_normal(self) -> None:
        for raw_priority in ("normal", "background", "unknown", ""):
            with self.subTest(priority=raw_priority):
                prepared = prepare_request(_request(priority=raw_priority))
                self.assertEqual(prepared["priority"], "normal")

    def test_routing_allows_fairness_key_only(self) -> None:
        prepared = prepare_request(_request(routing={"fairness_key": "session-1"}))

        self.assertEqual(prepared["routing"], {"fairness_key": "session-1"})

    def test_obsolete_routing_keys_are_rejected(self) -> None:
        for key in ("slot_affinity", "timeout_s"):
            with self.subTest(key=key):
                with self.assertRaises(AsrRequestError) as ctx:
                    prepare_request(_request(request_id=f"req-{key}", routing={key: 1}))
                self.assertEqual(ctx.exception.code, "ASR_UNKNOWN_ROUTING_KEY")
                self.assertEqual(ctx.exception.details["allowed_routing_keys"], ["fairness_key"])


if __name__ == "__main__":
    unittest.main()
