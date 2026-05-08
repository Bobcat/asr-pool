from __future__ import annotations

from types import SimpleNamespace
import unittest

from app.whisperx.server import PersistentWhisperxRunner


class _Segment:
    text = " hello"
    start = 0.1
    end = 0.8
    avg_logprob = -0.25
    compression_ratio = 1.2
    no_speech_prob = 0.1
    temperature = 0.0
    words = [SimpleNamespace(word="hello", start=0.1, end=0.5, probability=0.9)]


class _Info:
    language = "nl"
    language_probability = 0.92
    duration = 1.0
    duration_after_vad = 0.7


class _FakeFasterWhisperModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def transcribe(
        self,
        audio,
        *,
        condition_on_previous_text=None,
        vad_filter=None,
        beam_size=None,
        initial_prompt=None,
        language=None,
        chunk_length=None,
        vad_parameters=None,
        word_timestamps=None,
        max_new_tokens=None,
        hotwords=None,
        compression_ratio_threshold=None,
        log_prob_threshold=None,
        no_speech_threshold=None,
        language_detection_threshold=None,
        language_detection_segments=None,
    ):
        self.kwargs = {
            "condition_on_previous_text": condition_on_previous_text,
            "vad_filter": vad_filter,
            "beam_size": beam_size,
            "initial_prompt": initial_prompt,
            "language": language,
            "chunk_length": chunk_length,
            "vad_parameters": vad_parameters,
            "word_timestamps": word_timestamps,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "language_detection_threshold": language_detection_threshold,
            "language_detection_segments": language_detection_segments,
        }
        return [_Segment()], _Info()


class DirectFasterWhisperTests(unittest.TestCase):
    def test_transcribe_direct_maps_experiment_options_to_fw_kwargs(self) -> None:
        fw_model = _FakeFasterWhisperModel()
        runner = PersistentWhisperxRunner(cfg={"beam_size": 5})
        runner.asr_model = SimpleNamespace(model=fw_model)

        result, meta = runner._transcribe_direct_faster_whisper(
            audio_arr=[0.0] * 16000,
            language="nl",
            initial_prompt="context",
            beam_size_override=2,
            chunk_size_override=9,
            direct_options={
                "chunk_length": 4,
                "vad_filter": False,
                "vad_parameters": {"threshold": 0.4},
                "word_timestamps": True,
                "max_new_tokens": 64,
                "hotwords": "omniscripta realtime",
                "compression_ratio_threshold": 2.1,
                "log_prob_threshold": -0.7,
                "no_speech_threshold": 0.5,
                "language_detection_threshold": 0.6,
                "language_detection_segments": 2,
            },
        )

        self.assertEqual(result["language"], "nl")
        self.assertEqual(
            result["segments"],
            [
                {
                    "text": "hello",
                    "start": 0.1,
                    "end": 0.8,
                    "avg_logprob": -0.25,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.1,
                    "temperature": 0.0,
                    "words": [{"word": "hello", "start": 0.1, "end": 0.5, "probability": 0.9}],
                }
            ],
        )
        self.assertEqual(fw_model.kwargs["condition_on_previous_text"], False)
        self.assertEqual(fw_model.kwargs["vad_filter"], False)
        self.assertEqual(fw_model.kwargs["beam_size"], 2)
        self.assertEqual(fw_model.kwargs["initial_prompt"], "context")
        self.assertEqual(fw_model.kwargs["language"], "nl")
        self.assertEqual(fw_model.kwargs["chunk_length"], 4)
        self.assertEqual(fw_model.kwargs["vad_parameters"], {"threshold": 0.4})
        self.assertEqual(fw_model.kwargs["word_timestamps"], True)
        self.assertEqual(fw_model.kwargs["max_new_tokens"], 64)
        self.assertEqual(fw_model.kwargs["hotwords"], "omniscripta realtime")
        self.assertEqual(fw_model.kwargs["compression_ratio_threshold"], 2.1)
        self.assertEqual(fw_model.kwargs["log_prob_threshold"], -0.7)
        self.assertEqual(fw_model.kwargs["no_speech_threshold"], 0.5)
        self.assertEqual(fw_model.kwargs["language_detection_threshold"], 0.6)
        self.assertEqual(fw_model.kwargs["language_detection_segments"], 2)
        self.assertIn("chunk_length", meta["accepted_kwargs"])
        self.assertNotIn("chunk_size", meta["accepted_kwargs"])
        self.assertEqual(meta["chunk_size_override_applied"], False)
        self.assertEqual(meta["chunk_size_override_unsupported"], True)
        self.assertEqual(meta["chunk_length_override_applied"], True)
        self.assertEqual(
            meta["backend_metadata"],
            {"language_probability": 0.92, "duration": 1.0, "duration_after_vad": 0.7},
        )


if __name__ == "__main__":
    unittest.main()
