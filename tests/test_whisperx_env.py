from __future__ import annotations

import unittest
from unittest import mock

from app.whisperx.env import _load_server_config


class WhisperxEnvTests(unittest.TestCase):
    def test_load_server_config_uses_asr_backend_default(self) -> None:
        def fake_get_str(path: str, default: str = "") -> str:
            if path == "asr.backend":
                return "faster_whisper_direct"
            return str(default)

        with (
            mock.patch("app.whisperx.env.get_str", side_effect=fake_get_str),
            mock.patch("app.whisperx.env.get_int", side_effect=lambda _path, default=0, min_value=None: int(default)),
            mock.patch("app.whisperx.env.get_setting", return_value={}),
        ):
            cfg = _load_server_config()

        self.assertEqual(cfg["asr_backend"], "faster_whisper_direct")

    def test_load_server_config_rejects_invalid_asr_backend(self) -> None:
        def fake_get_str(path: str, default: str = "") -> str:
            if path == "asr.backend":
                return "bogus"
            return str(default)

        with (
            mock.patch("app.whisperx.env.get_str", side_effect=fake_get_str),
            mock.patch("app.whisperx.env.get_int", side_effect=lambda _path, default=0, min_value=None: int(default)),
            mock.patch("app.whisperx.env.get_setting", return_value={}),
        ):
            with self.assertRaises(ValueError) as ctx:
                _load_server_config()

        self.assertIn("Invalid asr.backend", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
