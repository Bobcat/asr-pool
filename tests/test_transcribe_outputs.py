from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest
import wave

from app.whisperx.transcribe import (
    ASR_BACKEND_WHISPERX,
    _extract_transcribe_request,
    _finalize_transcribe_phase,
)


def _write_wav(path: Path, *, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0\0" * frames)


class TranscribeOutputTests(unittest.TestCase):
    def test_extract_request_accepts_text_and_segments_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "input.wav"
            _write_wav(audio_path)

            ctx, err = _extract_transcribe_request(
                {
                    "request": {
                        "request_id": "req-json",
                        "audio": {"local_path": str(audio_path), "format": "wav"},
                        "outputs": {"text": True, "segments": True, "srt": False, "srt_inline": False},
                        "effective_options": {},
                    },
                    "work": {"whisperx_out_dir": str(tmp_path / "out")},
                }
            )

            self.assertIsNone(err)
            self.assertIsNotNone(ctx)
            self.assertEqual(ctx["outputs"]["text"], True)
            self.assertEqual(ctx["outputs"]["segments"], True)

    def test_finalize_returns_json_outputs_without_srt_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "input.wav"
            out_dir = tmp_path / "out"
            _write_wav(audio_path)

            def _unexpected_writer(*_args, **_kwargs):
                raise AssertionError("SRT writer should not be requested")

            response = _finalize_transcribe_phase(
                {"device": "cpu", "model": "tiny"},
                get_writer=_unexpected_writer,
                request_ctx={
                    "request_id": "req-json",
                    "effective_options": {},
                    "outputs": {"text": True, "segments": True, "srt": False, "srt_inline": False},
                    "audio": {"local_path": str(audio_path), "format": "wav"},
                    "local_path": audio_path,
                    "out_dir": out_dir,
                },
                runtime_ctx={
                    "selected_asr_backend": ASR_BACKEND_WHISPERX,
                    "selected_asr_backend_reason": "configured",
                    "language": "nl",
                    "beam_size_override": None,
                    "chunk_size_override": None,
                },
                phase_ctx={
                    "result": {"language": "nl"},
                    "aligned": {
                        "language": "nl",
                        "segments": [
                            {"text": "Hallo", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                            {"text": "wereld", "start": 0.4, "end": 0.8},
                        ],
                    },
                    "model_reused": True,
                    "direct_backend_meta": {},
                    "initial_prompt_applied": False,
                    "initial_prompt_unsupported": False,
                    "beam_size_override_applied": False,
                    "beam_size_override_unsupported": False,
                    "chunk_size_override_applied": False,
                    "chunk_size_override_unsupported": False,
                    "aligner_reused": None,
                    "align_skipped_missing_language": False,
                    "diarize_applied": False,
                    "diarizer_reused": None,
                    "transcribe_kwargs": {},
                },
                timings={},
                t_total=time.monotonic(),
                progress_path=None,
            )

            self.assertEqual(response["ok"], True)
            self.assertFalse(out_dir.exists())
            self.assertNotIn("artifacts", response["result"])
            self.assertNotIn("srt_text", response["result"])
            self.assertEqual(response["result"]["text"], "Hallo\nwereld")
            self.assertEqual(
                response["result"]["segments"],
                [
                    {"text": "Hallo", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                    {"text": "wereld", "start": 0.4, "end": 0.8},
                ],
            )
            self.assertEqual(response["result"]["language"], {"code": "nl"})
            self.assertEqual(response["result"]["backend_metadata"], {"backend": "whisperx"})
            self.assertEqual(response["runtime"]["segments_returned_count"], 2)
            self.assertEqual(response["warnings"], [])


if __name__ == "__main__":
    unittest.main()
