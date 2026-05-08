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

    def test_language_auto_normalizes_to_autodetect(self) -> None:
        for raw_language in ("auto", "AUTO", ""):
            with self.subTest(language=raw_language):
                prepared = prepare_request(_request(options={"language": raw_language}))
                self.assertIsNone(prepared["effective_options"].get("language"))

    def test_faster_whisper_experiment_options_are_normalized(self) -> None:
        prepared = prepare_request(
            _request(
                options={
                    "asr_backend": "FASTER_WHISPER_DIRECT",
                    "chunk_length": "4",
                    "vad_filter": "false",
                    "vad_parameters": {
                        "threshold": "0.42",
                        "min_speech_duration_ms": "0",
                        "speech_pad_ms": "80",
                        "unknown": "ignored",
                    },
                    "word_timestamps": "yes",
                    "max_new_tokens": "64",
                    "hotwords": "omniscripta realtime",
                    "compression_ratio_threshold": "2.1",
                    "log_prob_threshold": "-0.7",
                    "no_speech_threshold": "0.5",
                    "language_detection_threshold": "0.6",
                    "language_detection_segments": "2",
                },
            )
        )

        opts = prepared["effective_options"]
        self.assertEqual(opts["asr_backend"], "faster_whisper_direct")
        self.assertEqual(opts["chunk_length"], 4)
        self.assertEqual(opts["vad_filter"], False)
        self.assertEqual(
            opts["vad_parameters"],
            {"threshold": 0.42, "min_speech_duration_ms": 0, "speech_pad_ms": 80},
        )
        self.assertEqual(opts["word_timestamps"], True)
        self.assertEqual(opts["max_new_tokens"], 64)
        self.assertEqual(opts["hotwords"], "omniscripta realtime")
        self.assertEqual(opts["compression_ratio_threshold"], 2.1)
        self.assertEqual(opts["log_prob_threshold"], -0.7)
        self.assertEqual(opts["no_speech_threshold"], 0.5)
        self.assertEqual(opts["language_detection_threshold"], 0.6)
        self.assertEqual(opts["language_detection_segments"], 2)


if __name__ == "__main__":
    unittest.main()
