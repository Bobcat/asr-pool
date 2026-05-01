from __future__ import annotations

import io
import json
import time
import wave
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from app.asr.schema import ASR_SCHEMA_VERSION
from app.config import get_bool
from app.whisperx.env import _normalize_optional_language
from app.whisperx.io import _now_iso, _write_progress


ASR_BACKEND_WHISPERX = "whisperx"
ASR_BACKEND_FASTER_WHISPER_DIRECT = "faster_whisper_direct"
ASR_BACKEND_ALLOWED = {
  ASR_BACKEND_WHISPERX,
  ASR_BACKEND_FASTER_WHISPER_DIRECT,
}


def _audio_processed_ms_from_wave(path: Path, request_audio: dict[str, Any]) -> int | None:
  try:
    with wave.open(str(path), "rb") as wf:
      rate = int(wf.getframerate() or 0)
      frames = int(wf.getnframes() or 0)
      if rate > 0 and frames >= 0:
        return int(round((frames / float(rate)) * 1000.0))
  except Exception:
    pass
  try:
    val = request_audio.get("duration_ms")
    if val is not None:
      return int(val)
  except Exception:
    pass
  return None


def _wave_frame_count(path: Path) -> int | None:
  try:
    with wave.open(str(path), "rb") as wf:
      return int(wf.getnframes() or 0)
  except Exception:
    return None


def _is_wave_path(path: Path) -> bool:
  return str(path.suffix or "").strip().lower() in {".wav", ".wave"}


def _int_or_none(value: Any) -> int | None:
  try:
    return int(value)
  except Exception:
    return None


def _load_pcm16_wav_16khz_mono_or_none(request_ctx: dict[str, Any]) -> Any | None:
  audio = dict(request_ctx.get("audio") or {})
  audio_format = str(audio.get("format") or "").strip().lower()
  local_path = Path(str(request_ctx.get("local_path") or ""))
  if audio_format and audio_format not in {"wav", "wave"}:
    return None
  if not audio_format and not _is_wave_path(local_path):
    return None

  sample_rate_hz = _int_or_none(audio.get("sample_rate_hz"))
  if sample_rate_hz is not None and sample_rate_hz != 16000:
    return None
  channels = _int_or_none(audio.get("channels"))
  if channels is not None and channels != 1:
    return None

  try:
    with wave.open(str(local_path), "rb") as wf:
      if int(wf.getnchannels() or 0) != 1:
        return None
      if int(wf.getsampwidth() or 0) != 2:
        return None
      if int(wf.getframerate() or 0) != 16000:
        return None
      if str(wf.getcomptype() or "").upper() != "NONE":
        return None
      frames = int(wf.getnframes() or 0)
      if frames <= 0:
        return None
      raw = wf.readframes(frames)
  except Exception:
    return None
  if len(raw) != frames * 2:
    return None

  import numpy as np

  audio_arr = np.frombuffer(raw, dtype="<i2").astype(np.float32)
  audio_arr *= float(1.0 / 32768.0)
  return audio_arr


def _transcribe_error(
  *,
  request_id: str,
  effective_options: dict[str, Any],
  code: str,
  message: str,
  retryable: bool,
  details: dict[str, Any] | None = None,
  warnings: list[str] | None = None,
) -> dict[str, Any]:
  return {
    "schema_version": ASR_SCHEMA_VERSION,
    "request_id": str(request_id),
    "ok": False,
    "effective_options": dict(effective_options or {}),
    "error": {
      "code": str(code),
      "message": str(message),
      "retryable": bool(retryable),
      "details": dict(details or {}),
    },
    "warnings": list(warnings or []),
  }


def _extract_transcribe_request(envelope: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  request = dict(envelope.get("request") or {})
  work = dict(envelope.get("work") or {})
  req_id = str(request.get("request_id") or "")
  effective_options = dict(request.get("effective_options") or {})
  outputs = dict(request.get("outputs") or {})
  audio = dict(request.get("audio") or {})
  local_path = Path(str(audio.get("local_path") or ""))
  out_dir_raw = str(work.get("whisperx_out_dir") or "").strip()
  out_dir = Path(out_dir_raw) if out_dir_raw else Path()

  ctx = {
    "request": request,
    "work": work,
    "request_id": req_id,
    "effective_options": effective_options,
    "outputs": outputs,
    "audio": audio,
    "local_path": local_path,
    "out_dir_raw": out_dir_raw,
    "out_dir": out_dir,
  }

  if not local_path.exists():
    return None, _transcribe_error(
      request_id=req_id,
      effective_options=effective_options,
      code="ASR_INPUT_NOT_FOUND",
      message=f"ASR input not found: {local_path}",
      retryable=False,
      details={"local_path": str(local_path)},
    )

  unsupported_outputs = [k for k in ("text", "segments") if bool(outputs.get(k, False))]
  if unsupported_outputs:
    return None, _transcribe_error(
      request_id=req_id,
      effective_options=effective_options,
      code="ASR_UNSUPPORTED_OUTPUT",
      message="persistent ASR pool runner does not populate requested outputs",
      retryable=False,
      details={"requested_outputs": unsupported_outputs},
    )

  if not out_dir_raw:
    return None, _transcribe_error(
      request_id=req_id,
      effective_options=effective_options,
      code="ASR_OUTPUT_DIR_REQUIRED",
      message="Missing work.whisperx_out_dir",
      retryable=False,
    )

  try:
    file_size = int(local_path.stat().st_size)
  except Exception:
    file_size = -1
  if file_size == 0:
    return None, _transcribe_error(
      request_id=req_id,
      effective_options=effective_options,
      code="ASR_EMPTY_INPUT",
      message="ASR input audio file is empty",
      retryable=False,
      details={"local_path": str(local_path), "bytes": int(file_size)},
    )

  if _is_wave_path(local_path):
    frame_count = _wave_frame_count(local_path)
    if frame_count is None:
      return None, _transcribe_error(
        request_id=req_id,
        effective_options=effective_options,
        code="ASR_INVALID_AUDIO",
        message="ASR input audio could not be parsed as WAV",
        retryable=False,
        details={"local_path": str(local_path)},
      )
    if frame_count <= 0:
      return None, _transcribe_error(
        request_id=req_id,
        effective_options=effective_options,
        code="ASR_EMPTY_INPUT",
        message="ASR input audio contains no frames",
        retryable=False,
        details={"local_path": str(local_path), "frames": int(frame_count)},
      )

  return ctx, None


def _normalize_transcribe_runtime(
  *,
  effective_options: dict[str, Any],
  diarize_model: str | None,
  configured_asr_backend: str,
  configured_asr_backend_reason: str,
) -> dict[str, Any]:
  language = _normalize_optional_language(effective_options.get("language"))
  align_enabled = bool(effective_options.get("align_enabled", True))
  diarize_enabled = bool(effective_options.get("diarize_enabled", False))
  speaker_mode = str(effective_options.get("speaker_mode") or "none").strip().lower() or "none"
  min_speakers = effective_options.get("min_speakers")
  max_speakers = effective_options.get("max_speakers")
  initial_prompt = str(effective_options.get("initial_prompt") or "").strip() or None
  beam_size_override: int | None = None
  try:
    if effective_options.get("beam_size") is not None:
      beam_size_override = max(1, int(effective_options.get("beam_size")))
  except Exception:
    beam_size_override = None
  chunk_size_override: int | None = None
  try:
    if effective_options.get("chunk_size") is not None:
      chunk_size_override = max(1, int(effective_options.get("chunk_size")))
  except Exception:
    chunk_size_override = None
  if speaker_mode in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
    speaker_mode = "none"
  elif speaker_mode not in {"auto", "fixed"}:
    speaker_mode = "auto"
  asr_backend_override = str(effective_options.get("asr_backend") or "").strip().lower() or None
  if asr_backend_override not in ASR_BACKEND_ALLOWED:
    asr_backend_override = None
  aux_sensitive_mode = bool(align_enabled) or bool(diarize_enabled and speaker_mode != "none")
  if asr_backend_override is not None:
    selected_asr_backend = str(asr_backend_override)
    selected_asr_backend_reason = "request_override"
  else:
    selected_asr_backend = str(configured_asr_backend)
    selected_asr_backend_reason = str(configured_asr_backend_reason)
  return {
    "language": language,
    "align_enabled": align_enabled,
    "diarize_enabled": diarize_enabled,
    "speaker_mode": speaker_mode,
    "min_speakers": min_speakers,
    "max_speakers": max_speakers,
    "diarize_model": diarize_model,
    "initial_prompt": initial_prompt,
    "beam_size_override": beam_size_override,
    "chunk_size_override": chunk_size_override,
    "asr_backend_override": asr_backend_override,
    "aux_sensitive_mode": aux_sensitive_mode,
    "selected_asr_backend": selected_asr_backend,
    "selected_asr_backend_reason": selected_asr_backend_reason,
  }


def _build_transcribe_kwargs(
  cfg: dict[str, Any],
  *,
  language: str | None,
  chunk_size_override: int | None,
) -> dict[str, Any]:
  transcribe_kwargs: dict[str, Any] = {
    "batch_size": int(cfg.get("batch_size", 3) or 3),
    "chunk_size": int(cfg.get("chunk_size", 30) or 30),
    "print_progress": False,
    "verbose": False,
  }
  if language is not None:
    transcribe_kwargs["language"] = str(language)
  if chunk_size_override is not None:
    transcribe_kwargs["chunk_size"] = int(max(1, int(chunk_size_override)))
  return transcribe_kwargs


def _apply_transcribe_overrides(
  asr_model: Any,
  *,
  initial_prompt: str | None,
  beam_size_override: int | None,
) -> dict[str, bool]:
  flags = {
    "initial_prompt_applied": False,
    "initial_prompt_unsupported": False,
    "beam_size_override_applied": False,
    "beam_size_override_unsupported": False,
  }
  try:
    current_opts = getattr(asr_model, "options", None)
    if current_opts is not None:
      replace_kwargs: dict[str, Any] = {"initial_prompt": initial_prompt}
      if beam_size_override is not None:
        replace_kwargs["beam_size"] = int(beam_size_override)
      asr_model.options = dataclass_replace(current_opts, **replace_kwargs)
      flags["initial_prompt_applied"] = bool(initial_prompt is not None)
      flags["beam_size_override_applied"] = bool(beam_size_override is not None)
    elif initial_prompt is not None:
      flags["initial_prompt_unsupported"] = True
  except Exception:
    if initial_prompt is not None:
      flags["initial_prompt_unsupported"] = True
    if beam_size_override is not None:
      flags["beam_size_override_unsupported"] = True
  return flags


def _log_segment_debug(result: dict[str, Any]) -> None:
  try:
    segments = result.get("segments") or []
    import json as _json
    for idx, seg in enumerate(segments[:10]):
      seg_text = str(seg.get("text") or "").strip()
      seg_start = float(seg.get("start") or 0)
      seg_end = float(seg.get("end") or 0)
      seg_dur = round(seg_end - seg_start, 3)
      print(f"INFO seg_{idx} dur={seg_dur}s text={_json.dumps(seg_text, ensure_ascii=False)}", flush=True)
  except Exception:
    pass


def _log_transcribe_call_timing(
  *,
  request_id: str,
  selected_asr_backend: str,
  started_utc: str | None,
  finished_utc: str | None,
  duration_s: float | None,
) -> None:
  if started_utc is None or finished_utc is None or duration_s is None:
    return
  print(
    "ASR_TRANSCRIBE_CALL_TIMING "
    + json.dumps(
      {
        "request_id": request_id,
        "backend": (
          "faster_whisper_direct"
          if selected_asr_backend == ASR_BACKEND_FASTER_WHISPER_DIRECT
          else "whisperx"
        ),
        "start_utc": str(started_utc),
        "end_utc": str(finished_utc),
        "duration_s": float(duration_s),
      },
      ensure_ascii=False,
    ),
    flush=True,
  )


def _log_whisperx_transcribe_call_params(
  *,
  request_id: str,
  audio_path: Path,
  transcribe_kwargs: dict[str, Any],
  initial_prompt: str | None,
  beam_size_override: int | None,
) -> None:
  if not get_bool("whisperx.debug.log_transcribe_call_params", False):
    return
  print(
    "ASR_WHISPERX_TRANSCRIBE_CALL "
    + json.dumps(
      {
        "request_id": str(request_id),
        "audio_path": str(audio_path),
        "transcribe_kwargs": dict(transcribe_kwargs or {}),
        "initial_prompt": initial_prompt,
        "beam_size_override": beam_size_override,
      },
      ensure_ascii=False,
      sort_keys=True,
    ),
    flush=True,
  )


def _run_transcribe_phase(
  runner: Any,
  *,
  whisperx: Any,
  request_ctx: dict[str, Any],
  runtime_ctx: dict[str, Any],
  progress_path: Path | None,
  completed_timings: dict[str, Any] | None,
) -> dict[str, Any]:
  t0 = time.monotonic()
  transcribe_kwargs = _build_transcribe_kwargs(
    runner.cfg,
    language=runtime_ctx["language"],
    chunk_size_override=runtime_ctx["chunk_size_override"],
  )
  _write_progress(progress_path, stage="transcribe", timings=completed_timings)
  transcribe_call_started_utc: str | None = None
  transcribe_call_finished_utc: str | None = None
  transcribe_call_duration_s: float | None = None
  direct_backend_meta: dict[str, Any] = {}
  initial_prompt_applied = False
  initial_prompt_unsupported = False
  beam_override_applied = False
  beam_override_unsupported = False
  chunk_size_override_applied = bool(runtime_ctx["chunk_size_override"] is not None)
  chunk_size_override_unsupported = False
  load_audio_duration_s = 0.0

  if runtime_ctx["selected_asr_backend"] == ASR_BACKEND_WHISPERX:
    _log_whisperx_transcribe_call_params(
      request_id=str(request_ctx["request_id"]),
      audio_path=request_ctx["local_path"],
      transcribe_kwargs=transcribe_kwargs,
      initial_prompt=runtime_ctx["initial_prompt"],
      beam_size_override=runtime_ctx["beam_size_override"],
    )

  with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    load_audio_t0 = time.monotonic()
    audio_arr = _load_pcm16_wav_16khz_mono_or_none(request_ctx)
    if audio_arr is None:
      audio_arr = whisperx.load_audio(str(request_ctx["local_path"]))
    load_audio_duration_s = round(max(0.0, float(time.monotonic() - load_audio_t0)), 6)
    if runtime_ctx["selected_asr_backend"] == ASR_BACKEND_FASTER_WHISPER_DIRECT:
      transcribe_call_started_utc = _now_iso()
      transcribe_call_t0 = time.monotonic()
      try:
        result, direct_backend_meta = runner._transcribe_direct_faster_whisper(
          audio_arr=audio_arr,
          language=runtime_ctx["language"],
          initial_prompt=runtime_ctx["initial_prompt"],
          beam_size_override=runtime_ctx["beam_size_override"],
          chunk_size_override=runtime_ctx["chunk_size_override"],
        )
      finally:
        transcribe_call_finished_utc = _now_iso()
        transcribe_call_duration_s = round(max(0.0, float(time.monotonic() - transcribe_call_t0)), 6)
      initial_prompt_applied = bool(direct_backend_meta.get("initial_prompt_applied"))
      initial_prompt_unsupported = bool(direct_backend_meta.get("initial_prompt_unsupported"))
      beam_override_applied = bool(direct_backend_meta.get("beam_size_override_applied"))
      beam_override_unsupported = bool(direct_backend_meta.get("beam_size_override_unsupported"))
      chunk_size_override_applied = bool(direct_backend_meta.get("chunk_size_override_applied"))
      chunk_size_override_unsupported = bool(direct_backend_meta.get("chunk_size_override_unsupported"))
    else:
      override_flags = _apply_transcribe_overrides(
        runner.asr_model,
        initial_prompt=runtime_ctx["initial_prompt"],
        beam_size_override=runtime_ctx["beam_size_override"],
      )
      initial_prompt_applied = bool(override_flags["initial_prompt_applied"])
      initial_prompt_unsupported = bool(override_flags["initial_prompt_unsupported"])
      beam_override_applied = bool(override_flags["beam_size_override_applied"])
      beam_override_unsupported = bool(override_flags["beam_size_override_unsupported"])
      transcribe_call_started_utc = _now_iso()
      transcribe_call_t0 = time.monotonic()
      try:
        result = runner.asr_model.transcribe(audio_arr, **transcribe_kwargs)
      finally:
        transcribe_call_finished_utc = _now_iso()
        transcribe_call_duration_s = round(max(0.0, float(time.monotonic() - transcribe_call_t0)), 6)
    _log_segment_debug(result)

  _log_transcribe_call_timing(
    request_id=str(request_ctx["request_id"]),
    selected_asr_backend=str(runtime_ctx["selected_asr_backend"]),
    started_utc=transcribe_call_started_utc,
    finished_utc=transcribe_call_finished_utc,
    duration_s=transcribe_call_duration_s,
  )

  transcribe_duration_s = round(max(0.0, float(time.monotonic() - t0)), 6)
  transcribe_overhead_s = round(
    max(
      0.0,
      float(transcribe_duration_s - (transcribe_call_duration_s if transcribe_call_duration_s is not None else 0.0)),
    ),
    6,
  )

  return {
    "result": result,
    "audio_arr": audio_arr,
    "transcribe_kwargs": transcribe_kwargs,
    "direct_backend_meta": direct_backend_meta,
    "initial_prompt_applied": initial_prompt_applied,
    "initial_prompt_unsupported": initial_prompt_unsupported,
    "beam_size_override_applied": beam_override_applied,
    "beam_size_override_unsupported": beam_override_unsupported,
    "chunk_size_override_applied": chunk_size_override_applied,
    "chunk_size_override_unsupported": chunk_size_override_unsupported,
    "load_audio_s": load_audio_duration_s,
    "transcribe_s": transcribe_duration_s,
    "transcribe_call_s": transcribe_call_duration_s,
    "transcribe_overhead_s": transcribe_overhead_s,
  }


def _run_alignment_phase(
  runner: Any,
  *,
  whisperx: Any,
  result: dict[str, Any],
  audio_arr: Any,
  runtime_ctx: dict[str, Any],
  progress_path: Path | None,
  completed_timings: dict[str, Any] | None,
) -> dict[str, Any]:
  t0 = time.monotonic()
  align_language = _normalize_optional_language(result.get("language"))
  if align_language is None:
    align_language = runtime_ctx["language"]
  aligner_reused = None
  aligner_load_s = 0.0
  align_skipped_missing_language = False

  if bool(runtime_ctx["align_enabled"]) and align_language is not None:
    _write_progress(progress_path, stage="align", timings=completed_timings)
    aligner, align_meta, aligner_reused, aligner_load_s = runner._ensure_aligner(language=align_language)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      if aligner is not None and len(result.get("segments") or []) > 0:
        aligned = whisperx.align(
          result["segments"],
          aligner,
          align_meta,
          audio_arr,
          str(runner.cfg.get("device", "cuda") or "cuda"),
          return_char_alignments=False,
          print_progress=False,
        )
      else:
        aligned = {"segments": result.get("segments") or []}
    aligned["language"] = str((align_meta or {}).get("language") or align_language)
  else:
    aligned = {"segments": result.get("segments") or []}
    aligned["language"] = align_language
    if bool(runtime_ctx["align_enabled"]) and align_language is None:
      align_skipped_missing_language = True

  out = {
    "aligned": aligned,
    "align_s": round(max(0.0, float(time.monotonic() - t0)), 6),
    "aligner_reused": aligner_reused,
    "align_skipped_missing_language": align_skipped_missing_language,
  }
  if aligner_load_s > 0:
    out["aligner_load_s"] = round(float(aligner_load_s), 6)
  return out


def _run_diarization_phase(
  runner: Any,
  *,
  whisperx: Any,
  aligned: dict[str, Any],
  local_path: Path,
  runtime_ctx: dict[str, Any],
  progress_path: Path | None,
  completed_timings: dict[str, Any] | None,
) -> dict[str, Any]:
  diarize_applied = False
  diarizer_reused: bool | None = None
  diarizer_load_s = 0.0
  t0 = time.monotonic()

  if bool(runtime_ctx["diarize_enabled"]) and str(runtime_ctx["speaker_mode"]) != "none":
    _write_progress(progress_path, stage="diarize", timings=completed_timings)
    diarize_kwargs: dict[str, Any] = {}
    if str(runtime_ctx["speaker_mode"]) == "fixed":
      if runtime_ctx["min_speakers"] is not None:
        try:
          diarize_kwargs["min_speakers"] = int(runtime_ctx["min_speakers"])
        except Exception:
          pass
      if runtime_ctx["max_speakers"] is not None:
        try:
          diarize_kwargs["max_speakers"] = int(runtime_ctx["max_speakers"])
        except Exception:
          pass
    try:
      diarize_pipe, diarizer_reused, diarizer_load_s = runner._ensure_diarizer(
        diarize_model=runtime_ctx["diarize_model"],
      )
      with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        diarize_df = diarize_pipe(str(local_path), **diarize_kwargs)
        aligned = whisperx.assign_word_speakers(diarize_df, aligned)
      diarize_applied = True
    except Exception:
      diarize_applied = False

  out = {
    "aligned": aligned,
    "diarize_applied": diarize_applied,
    "diarizer_reused": diarizer_reused,
    "diarize_s": round(max(0.0, float(time.monotonic() - t0)), 6),
  }
  if diarizer_load_s > 0:
    out["diarizer_load_s"] = round(float(diarizer_load_s), 6)
  return out


def _finalize_transcribe_phase(
  cfg: dict[str, Any],
  *,
  get_writer: Any,
  request_ctx: dict[str, Any],
  runtime_ctx: dict[str, Any],
  phase_ctx: dict[str, Any],
  timings: dict[str, float],
  t_total: float,
  progress_path: Path | None,
) -> dict[str, Any]:
  _write_progress(progress_path, stage="finalize", timings=timings)
  t0 = time.monotonic()
  out_dir = request_ctx["out_dir"]
  local_path = request_ctx["local_path"]
  out_dir.mkdir(parents=True, exist_ok=True)
  writer = get_writer("srt", str(out_dir))
  writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
  with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    writer(phase_ctx["aligned"], str(local_path), writer_args)
  timings["finalize_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)

  srt_path = out_dir / f"{local_path.stem}.srt"
  if not srt_path.exists():
    srts = sorted(out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime)
    if not srts:
      return _transcribe_error(
        request_id=request_ctx["request_id"],
        effective_options=request_ctx["effective_options"],
        code="ASR_OUTPUT_MISSING",
        message=f"No .srt produced in {out_dir}",
        retryable=True,
        details={"out_dir": str(out_dir)},
      )
    srt_path = srts[-1]

  timings["total_s"] = round(max(0.0, float(time.monotonic() - t_total)), 6)
  audio_ms = _audio_processed_ms_from_wave(local_path, request_ctx["audio"])

  result_obj: dict[str, Any] = {
    "artifacts": {
      "srt_path": str(srt_path),
    },
  }
  if audio_ms is not None:
    result_obj["audio_processed_ms"] = int(audio_ms)
  if bool(request_ctx["outputs"].get("srt_inline", False)):
    try:
      result_obj["srt_text"] = srt_path.read_text(encoding="utf-8")
    except Exception:
      pass

  segments_returned_count = int(len(phase_ctx["result"].get("segments") or []))
  runtime = {
    "backend": (
      "faster_whisper_direct"
      if runtime_ctx["selected_asr_backend"] == ASR_BACKEND_FASTER_WHISPER_DIRECT
      else "whisperx"
    ),
    "runner_kind": "persistent_local",
    "runner_reused": bool(phase_ctx["model_reused"]),
    "device": str(cfg.get("device") or ""),
    "model": str(cfg.get("model") or ""),
    "asr_backend_selected": str(runtime_ctx["selected_asr_backend"]),
    "asr_backend_reason": str(runtime_ctx["selected_asr_backend_reason"]),
    "segments_returned_count": int(segments_returned_count),
    "effective_batch_size": int(phase_ctx["transcribe_kwargs"].get("batch_size") or 0),
    "effective_chunk_size": int(phase_ctx["transcribe_kwargs"].get("chunk_size") or 0),
    "diarize_applied": bool(phase_ctx["diarize_applied"]),
    "initial_prompt_applied": bool(phase_ctx["initial_prompt_applied"]),
    "beam_size_override_applied": bool(phase_ctx["beam_size_override_applied"]),
    "chunk_size_override_applied": bool(phase_ctx["chunk_size_override_applied"]),
    "beam_size_override": (
      int(runtime_ctx["beam_size_override"])
      if runtime_ctx["beam_size_override"] is not None
      else None
    ),
    "chunk_size_override": (
      int(runtime_ctx["chunk_size_override"])
      if runtime_ctx["chunk_size_override"] is not None
      else None
    ),
  }
  if phase_ctx["aligner_reused"] is not None:
    runtime["aligner_reused"] = bool(phase_ctx["aligner_reused"])
  if phase_ctx["diarizer_reused"] is not None:
    runtime["diarizer_reused"] = bool(phase_ctx["diarizer_reused"])
  if phase_ctx["direct_backend_meta"]:
    runtime["direct_backend_meta"] = dict(phase_ctx["direct_backend_meta"])

  warnings: list[str] = []
  if bool(phase_ctx["align_skipped_missing_language"]):
    warnings.append("align_skipped_missing_language")
  if bool(phase_ctx["initial_prompt_unsupported"]):
    warnings.append("initial_prompt_unsupported_by_asr_pipeline")
  if bool(phase_ctx["beam_size_override_unsupported"]):
    warnings.append("beam_size_override_unsupported_by_asr_pipeline")
  if bool(phase_ctx["chunk_size_override_unsupported"]):
    warnings.append("chunk_size_override_unsupported_by_asr_pipeline")
  if runtime_ctx["selected_asr_backend"] == ASR_BACKEND_FASTER_WHISPER_DIRECT:
    warnings.append("asr_backend_faster_whisper_direct_experimental")

  return {
    "schema_version": ASR_SCHEMA_VERSION,
    "request_id": request_ctx["request_id"],
    "ok": True,
    "effective_options": request_ctx["effective_options"],
    "result": result_obj,
    "timings": timings,
    "runtime": runtime,
    "warnings": warnings,
  }
