from __future__ import annotations

from typing import Any


class AsrOptionsError(ValueError):
  def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.code = str(code)
    self.details = dict(details or {})


_KNOWN_OPTION_KEYS = {
  "language",
  "align_enabled",
  "diarize_enabled",
  "speaker_mode",
  "min_speakers",
  "max_speakers",
  "beam_size",
  "chunk_size",
  "chunk_length",
  "asr_backend",
  "initial_prompt",
  "vad_filter",
  "vad_parameters",
  "word_timestamps",
  "max_new_tokens",
  "hotwords",
  "compression_ratio_threshold",
  "log_prob_threshold",
  "no_speech_threshold",
  "language_detection_threshold",
  "language_detection_segments",
}


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  s = str(value or "").strip().lower()
  return s in {"1", "true", "yes", "on", "y"}


def _as_positive_int_or_none(value: Any) -> int | None:
  try:
    return max(1, int(value))
  except Exception:
    return None


def _as_nonnegative_int_or_none(value: Any) -> int | None:
  try:
    return max(0, int(value))
  except Exception:
    return None


def _as_float_or_none(value: Any) -> float | None:
  try:
    return float(value)
  except Exception:
    return None


def _clean_vad_parameters(value: Any) -> dict[str, Any] | None:
  if not isinstance(value, dict):
    return None
  allowed = {
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "max_speech_duration_s",
    "min_silence_duration_ms",
    "speech_pad_ms",
    "min_silence_at_max_speech",
    "use_max_poss_sil_at_max_speech",
  }
  out: dict[str, Any] = {}
  for raw_key, raw_value in value.items():
    key = str(raw_key or "").strip()
    if key not in allowed or raw_value is None:
      continue
    if key == "use_max_poss_sil_at_max_speech":
      out[key] = _as_bool(raw_value)
      continue
    if key in {"threshold", "neg_threshold", "max_speech_duration_s"}:
      parsed_float = _as_float_or_none(raw_value)
      if parsed_float is not None:
        out[key] = float(parsed_float)
      continue
    parsed_int = _as_nonnegative_int_or_none(raw_value)
    if parsed_int is not None:
      out[key] = int(parsed_int)
  return out or None


def normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
  opts = dict(options or {})
  for key in list(opts.keys()):
    if key not in _KNOWN_OPTION_KEYS:
      raise AsrOptionsError(
        "ASR_UNKNOWN_OPTION",
        f"Unknown ASR option: {key}",
        details={"option": key},
      )

  resolved: dict[str, Any] = {
    "align_enabled": False,
    "diarize_enabled": False,
    "speaker_mode": "none",
  }
  for key, value in opts.items():
    if value is None:
      continue
    resolved[key] = value

  if "language" in resolved and resolved["language"] is not None:
    language = str(resolved["language"]).strip().lower()
    resolved["language"] = None if language in {"", "auto"} else language

  if "speaker_mode" in resolved and resolved["speaker_mode"] is not None:
    speaker_mode = str(resolved["speaker_mode"]).strip().lower()
    if speaker_mode in {"off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
      speaker_mode = "none"
    if speaker_mode not in {"none", "auto", "fixed"}:
      speaker_mode = "auto"
    resolved["speaker_mode"] = speaker_mode

  if "align_enabled" in resolved and resolved["align_enabled"] is not None:
    resolved["align_enabled"] = _as_bool(resolved["align_enabled"])
  if "diarize_enabled" in resolved and resolved["diarize_enabled"] is not None:
    resolved["diarize_enabled"] = _as_bool(resolved["diarize_enabled"])

  for key in ("min_speakers", "max_speakers"):
    if key in resolved and resolved[key] is not None:
      try:
        resolved[key] = int(resolved[key])
      except Exception:
        resolved[key] = None

  if "beam_size" in resolved and resolved["beam_size"] is not None:
    try:
      resolved["beam_size"] = max(1, int(resolved["beam_size"]))
    except Exception:
      resolved["beam_size"] = 5

  if "chunk_size" in resolved and resolved["chunk_size"] is not None:
    resolved["chunk_size"] = _as_positive_int_or_none(resolved["chunk_size"])

  if "chunk_length" in resolved and resolved["chunk_length"] is not None:
    resolved["chunk_length"] = _as_positive_int_or_none(resolved["chunk_length"])

  if "asr_backend" in resolved and resolved["asr_backend"] is not None:
    asr_backend = str(resolved["asr_backend"]).strip().lower()
    if asr_backend not in {"whisperx", "faster_whisper_direct"}:
      asr_backend = None
    resolved["asr_backend"] = asr_backend

  if "initial_prompt" in resolved and resolved["initial_prompt"] is not None:
    resolved["initial_prompt"] = str(resolved["initial_prompt"])

  for key in ("vad_filter", "word_timestamps"):
    if key in resolved and resolved[key] is not None:
      resolved[key] = _as_bool(resolved[key])

  if "vad_parameters" in resolved and resolved["vad_parameters"] is not None:
    resolved["vad_parameters"] = _clean_vad_parameters(resolved["vad_parameters"])

  if "max_new_tokens" in resolved and resolved["max_new_tokens"] is not None:
    resolved["max_new_tokens"] = _as_positive_int_or_none(resolved["max_new_tokens"])

  if "hotwords" in resolved and resolved["hotwords"] is not None:
    hotwords = str(resolved["hotwords"]).strip()
    resolved["hotwords"] = hotwords or None

  for key in ("compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold", "language_detection_threshold"):
    if key in resolved and resolved[key] is not None:
      resolved[key] = _as_float_or_none(resolved[key])

  if "language_detection_segments" in resolved and resolved["language_detection_segments"] is not None:
    resolved["language_detection_segments"] = _as_positive_int_or_none(resolved["language_detection_segments"])

  return resolved
