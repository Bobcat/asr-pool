from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from app.asr.schema import ASR_SCHEMA_VERSION
from app.whisperx.env import _normalize_optional_language
from app.whisperx.io import _read_json, _write_json_atomic, _write_progress
from app.whisperx.imports import _apply_torch_thread_tuning, _as_positive_int, _cleanup_torch
from app.whisperx.transcribe import (
  ASR_BACKEND_ALLOWED,
  _extract_transcribe_request,
  _finalize_transcribe_phase,
  _normalize_transcribe_runtime,
  _run_alignment_phase,
  _run_diarization_phase,
  _run_transcribe_phase,
)



class PersistentWhisperxRunner:
  def __init__(self, cfg: dict[str, Any]) -> None:
    self.cfg = dict(cfg or {})
    self.whisperx = None
    self.torch = None
    self.get_writer = None
    self.asr_model = None
    self.asr_key = None
    self.aligners: dict[tuple[str | None, str | None], tuple[Any, dict[str, Any]]] = {}
    self.diarizers: dict[tuple[str | None, str], Any] = {}
    self._imported = False

  def _import_deps(self) -> None:
    if self._imported:
      return
    import whisperx  # type: ignore
    import torch  # type: ignore
    from whisperx.utils import get_writer  # type: ignore

    self.whisperx = whisperx
    self.torch = torch
    self.get_writer = get_writer
    self._imported = True

    torch_num_threads = _as_positive_int(self.cfg.get("torch_num_threads"))
    torch_num_interop_threads = _as_positive_int(self.cfg.get("torch_num_interop_threads"))
    try:
      _apply_torch_thread_tuning(
        torch,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
      )
    except Exception:
      pass

  def _asr_cache_key(self, *, language: str | None) -> tuple[Any, ...]:
    # Keep one language-agnostic warm ASR model per runner slot.
    # Per-call language hints are handled at transcribe() time.
    return (
      str(self.cfg.get("model") or "large-v3"),
      str(self.cfg.get("device") or "cuda"),
      str(self.cfg.get("compute_type") or "float16"),
      "__language_agnostic__",
      int(self.cfg.get("beam_size", 5) or 5),
      int(self.cfg.get("chunk_size", 30) or 30),
    )

  def _resolve_default_asr_backend(self) -> tuple[str, str]:
    raw = str(self.cfg.get("low_latency_backend") or "").strip().lower()
    if raw in ASR_BACKEND_ALLOWED:
      return raw, "configured"
    raise RuntimeError(f"Invalid default ASR backend configuration: {raw!r}")

  def _transcribe_direct_faster_whisper(
    self,
    *,
    audio_arr: Any,
    language: str | None,
    initial_prompt: str | None,
    beam_size_override: int | None,
    chunk_size_override: int | None,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    if self.asr_model is None:
      raise RuntimeError("ASR model not loaded")
    fw_model = getattr(self.asr_model, "model", None)
    if fw_model is None:
      raise RuntimeError("ASR model has no direct faster-whisper backend")

    requested_kwargs: dict[str, Any] = {
      "condition_on_previous_text": False,
      "vad_filter": True,
      "beam_size": int(beam_size_override if beam_size_override is not None else int(self.cfg.get("beam_size", 5) or 5)),
      "chunk_size": int(chunk_size_override) if chunk_size_override is not None else None,
      "initial_prompt": initial_prompt,
    }
    if language is not None:
      requested_kwargs["language"] = str(language)
    try:
      sig = inspect.signature(fw_model.transcribe)
      accepted = set(sig.parameters.keys())
    except Exception:
      accepted = set()

    call_kwargs: dict[str, Any] = {
      k: v
      for k, v in requested_kwargs.items()
      if v is not None and (not accepted or k in accepted)
    }
    dropped_kwargs = [
      k
      for k, v in requested_kwargs.items()
      if v is not None and accepted and k not in accepted
    ]

    # Experimental direct path note:
    # This bypasses WhisperX's transcribe pipeline conveniences before decode:
    # - no WhisperX VAD preprocess + merge_chunks
    # - no WhisperX tokenizer/language lifecycle in transcribe()
    # - no WhisperX postprocess conventions (it returns richer segment objects that we flatten)
    fw_output = fw_model.transcribe(audio_arr, **call_kwargs)
    if isinstance(fw_output, tuple) and len(fw_output) >= 2:
      fw_segments_iter, fw_info = fw_output[0], fw_output[1]
    else:
      fw_segments_iter, fw_info = fw_output, None

    audio_duration_s = 0.0
    try:
      audio_duration_s = float(len(audio_arr)) / 16000.0
    except Exception:
      audio_duration_s = 0.0

    segments: list[dict[str, Any]] = []
    for seg in fw_segments_iter:
      try:
        raw_text = str(getattr(seg, "text", "") or "").strip()
        if not raw_text:
          continue
        t0 = float(getattr(seg, "start", 0.0) or 0.0)
        t1 = float(getattr(seg, "end", t0) or t0)
        if audio_duration_s > 0.0:
          t0 = max(0.0, min(t0, audio_duration_s))
          t1 = max(t0, min(t1, audio_duration_s))
        segments.append(
          {
            "text": raw_text,
            "start": round(float(t0), 3),
            "end": round(float(t1), 3),
          }
        )
      except Exception:
        continue

    out_language = _normalize_optional_language(language)
    try:
      if fw_info is not None:
        fw_language = _normalize_optional_language(getattr(fw_info, "language", None))
        if fw_language is not None:
          out_language = fw_language
    except Exception:
      pass

    meta = {
      "accepted_kwargs": sorted(call_kwargs.keys()),
      "dropped_kwargs": sorted(dropped_kwargs),
      "segments_returned_count": int(len(segments)),
      "initial_prompt_applied": bool(initial_prompt) and ("initial_prompt" in call_kwargs),
      "initial_prompt_unsupported": bool(initial_prompt) and ("initial_prompt" not in call_kwargs),
      "beam_size_override_applied": (beam_size_override is not None) and ("beam_size" in call_kwargs),
      "beam_size_override_unsupported": (beam_size_override is not None) and ("beam_size" not in call_kwargs),
      "chunk_size_override_applied": (chunk_size_override is not None) and ("chunk_size" in call_kwargs),
      "chunk_size_override_unsupported": (chunk_size_override is not None) and ("chunk_size" not in call_kwargs),
    }
    return {"segments": segments, "language": out_language}, meta

  def _ensure_asr_model(self, *, language: str | None) -> tuple[bool, float]:
    self._import_deps()
    key = self._asr_cache_key(language=language)
    if self.asr_model is not None and self.asr_key == key:
      return True, 0.0
    t0 = time.monotonic()
    if self.asr_model is not None:
      try:
        del self.asr_model
      except Exception:
        pass
      self.asr_model = None
      self.asr_key = None
      try:
        _cleanup_torch(self.torch)
      except Exception:
        pass

    whisperx = self.whisperx
    assert whisperx is not None
    self.asr_model = whisperx.load_model(
      str(self.cfg.get("model", "large-v3") or "large-v3"),
      device=str(self.cfg.get("device", "cuda") or "cuda"),
      compute_type=str(self.cfg.get("compute_type", "float16") or "float16"),
      language=None,
      asr_options={"beam_size": int(self.cfg.get("beam_size", 5) or 5)},
      vad_options={"chunk_size": int(self.cfg.get("chunk_size", 30) or 30)},
    )
    self.asr_key = key
    return False, max(0.0, float(time.monotonic() - t0))

  def _ensure_aligner(self, *, language: str) -> tuple[Any, dict[str, Any], bool, float]:
    self._import_deps()
    align_model = str(self.cfg.get("align_model") or "").strip() or None
    key = (language, align_model)
    if key in self.aligners:
      aligner, meta = self.aligners[key]
      return aligner, dict(meta or {}), True, 0.0
    t0 = time.monotonic()
    whisperx = self.whisperx
    assert whisperx is not None
    aligner, meta = whisperx.load_align_model(
      language,
      str(self.cfg.get("device", "cuda") or "cuda"),
      model_name=align_model,
    )
    self.aligners[key] = (aligner, dict(meta or {}))
    return aligner, dict(meta or {}), False, max(0.0, float(time.monotonic() - t0))

  def _ensure_diarizer(self, *, diarize_model: str | None) -> tuple[Any, bool, float]:
    self._import_deps()
    device = str(self.cfg.get("device", "cuda") or "cuda")
    key = (diarize_model, device)
    if key in self.diarizers:
      return self.diarizers[key], True, 0.0
    t0 = time.monotonic()
    from whisperx.diarize import DiarizationPipeline  # type: ignore

    token = str(os.getenv("HF_TOKEN") or "").strip() or None
    diarize_kwargs: dict[str, Any] = {
      "model_name": diarize_model,
      "device": device,
    }
    # WhisperX versions differ on auth kwarg name:
    # - older: use_auth_token=
    # - newer: token=
    if token is not None:
      try:
        diarize_pipe = DiarizationPipeline(
          **diarize_kwargs,
          use_auth_token=token,
        )
      except TypeError:
        diarize_pipe = DiarizationPipeline(
          **diarize_kwargs,
          token=token,
        )
    else:
      diarize_pipe = DiarizationPipeline(**diarize_kwargs)
    self.diarizers[key] = diarize_pipe
    return diarize_pipe, False, max(0.0, float(time.monotonic() - t0))

  def _release_aux_models(self) -> None:
    for _k, (aligner, _meta) in list(self.aligners.items()):
      try:
        to_cpu = getattr(aligner, "cpu", None)
        if callable(to_cpu):
          try:
            to_cpu()
          except Exception:
            pass
        to_dev = getattr(aligner, "to", None)
        if callable(to_dev):
          try:
            to_dev("cpu")
          except Exception:
            pass
        del aligner
      except Exception:
        pass
    self.aligners.clear()
    for _k, diarize_pipe in list(self.diarizers.items()):
      try:
        model_obj = getattr(diarize_pipe, "model", None)
        if model_obj is not None:
          to_cpu = getattr(model_obj, "cpu", None)
          if callable(to_cpu):
            try:
              to_cpu()
            except Exception:
              pass
          to_dev = getattr(model_obj, "to", None)
          if callable(to_dev):
            try:
              to_dev("cpu")
            except Exception:
              pass
        pipe_to = getattr(diarize_pipe, "to", None)
        if callable(pipe_to):
          try:
            pipe_to("cpu")
          except Exception:
            pass
        del diarize_pipe
      except Exception:
        pass
    self.diarizers.clear()
    try:
      if self.torch is not None:
        _cleanup_torch(self.torch)
    except Exception:
      pass
    try:
      gc.collect()
    except Exception:
      pass

  def prewarm(self, *, language: str | None, align_enabled: bool = False) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    resolved_language = _normalize_optional_language(language)
    model_reused, prepare_s = self._ensure_asr_model(language=resolved_language)
    timings["prepare_s"] = round(float(prepare_s), 6)
    aligner_reused: bool | None = None
    if bool(align_enabled) and resolved_language is not None:
      _aligner, _meta, aligner_reused, aligner_load_s = self._ensure_aligner(language=resolved_language)
      timings["aligner_prepare_s"] = round(float(max(0.0, aligner_load_s)), 6)
    timings["total_s"] = round(float(max(0.0, time.monotonic() - t0)), 6)
    return {
      "ok": True,
      "language": resolved_language,
      "align_enabled": bool(align_enabled),
      "runner_reused": bool(model_reused),
      "aligner_reused": (None if aligner_reused is None else bool(aligner_reused)),
      "timings": timings,
      "runtime": {
        "backend": "whisperx",
        "runner_kind": "persistent_local",
        "device": str(self.cfg.get("device") or ""),
        "model": str(self.cfg.get("model") or ""),
      },
    }

  def _prepare_transcribe_models(
    self,
    *,
    language: str | None,
    progress_path: Path | None,
  ) -> tuple[bool, float]:
    _write_progress(progress_path, stage="prepare")
    return self._ensure_asr_model(language=language)

  def transcribe(self, envelope: dict[str, Any], *, progress_path: Path | None = None) -> dict[str, Any]:
    aux_sensitive_mode = False
    timings: dict[str, float] = {}
    try:
      request_ctx, error = _extract_transcribe_request(envelope)
      if error is not None:
        return error
      assert request_ctx is not None

      configured_asr_backend, configured_asr_backend_reason = self._resolve_default_asr_backend()
      runtime_ctx = _normalize_transcribe_runtime(
        effective_options=request_ctx["effective_options"],
        diarize_model=(str(self.cfg.get("diarize_model") or "").strip() or None),
        configured_asr_backend=configured_asr_backend,
        configured_asr_backend_reason=configured_asr_backend_reason,
      )
      aux_sensitive_mode = bool(runtime_ctx["aux_sensitive_mode"])
      t_total = time.monotonic()
      self._import_deps()
      whisperx = self.whisperx
      get_writer = self.get_writer
      assert whisperx is not None and get_writer is not None and self.torch is not None

      if aux_sensitive_mode:
        # Requests that use align/diarize can retain extra aux-model VRAM.
        # Release before request to keep a stable baseline.
        self._release_aux_models()

      model_reused, prepare_s = self._prepare_transcribe_models(
        language=runtime_ctx["language"],
        progress_path=progress_path,
      )
      timings["prepare_s"] = round(float(prepare_s), 6)

      transcribe_phase = _run_transcribe_phase(
        self,
        whisperx=whisperx,
        request_ctx=request_ctx,
        runtime_ctx=runtime_ctx,
        progress_path=progress_path,
        completed_timings=timings,
      )
      timings["transcribe_s"] = float(transcribe_phase["transcribe_s"])
      if transcribe_phase["transcribe_call_s"] is not None:
        timings["transcribe_call_s"] = round(float(transcribe_phase["transcribe_call_s"]), 6)

      alignment_phase = _run_alignment_phase(
        self,
        whisperx=whisperx,
        result=transcribe_phase["result"],
        audio_arr=transcribe_phase["audio_arr"],
        runtime_ctx=runtime_ctx,
        progress_path=progress_path,
        completed_timings=timings,
      )
      timings["align_s"] = float(alignment_phase["align_s"])
      if "aligner_load_s" in alignment_phase:
        timings["aligner_load_s"] = float(alignment_phase["aligner_load_s"])

      diarize_phase = _run_diarization_phase(
        self,
        whisperx=whisperx,
        aligned=alignment_phase["aligned"],
        local_path=request_ctx["local_path"],
        runtime_ctx=runtime_ctx,
        progress_path=progress_path,
        completed_timings=timings,
      )
      timings["diarize_s"] = float(diarize_phase["diarize_s"])
      if "diarizer_load_s" in diarize_phase:
        timings["diarizer_load_s"] = float(diarize_phase["diarizer_load_s"])

      phase_ctx = {
        "result": transcribe_phase["result"],
        "aligned": diarize_phase["aligned"],
        "transcribe_kwargs": transcribe_phase["transcribe_kwargs"],
        "model_reused": bool(model_reused),
        "direct_backend_meta": dict(transcribe_phase["direct_backend_meta"]),
        "initial_prompt_applied": bool(transcribe_phase["initial_prompt_applied"]),
        "initial_prompt_unsupported": bool(transcribe_phase["initial_prompt_unsupported"]),
        "beam_size_override_applied": bool(transcribe_phase["beam_size_override_applied"]),
        "beam_size_override_unsupported": bool(transcribe_phase["beam_size_override_unsupported"]),
        "chunk_size_override_applied": bool(transcribe_phase["chunk_size_override_applied"]),
        "chunk_size_override_unsupported": bool(transcribe_phase["chunk_size_override_unsupported"]),
        "aligner_reused": alignment_phase["aligner_reused"],
        "align_skipped_missing_language": bool(alignment_phase["align_skipped_missing_language"]),
        "diarize_applied": bool(diarize_phase["diarize_applied"]),
        "diarizer_reused": diarize_phase["diarizer_reused"],
      }
      return _finalize_transcribe_phase(
        self.cfg,
        get_writer=get_writer,
        request_ctx=request_ctx,
        runtime_ctx=runtime_ctx,
        phase_ctx=phase_ctx,
        timings=timings,
        t_total=t_total,
        progress_path=progress_path,
      )
    finally:
      _write_progress(progress_path, stage="done", timings=timings)
      if aux_sensitive_mode:
        # Keep inter-request VRAM baseline low when auxiliary models are used.
        self._release_aux_models()

  def shutdown(self) -> None:
    try:
      if self.asr_model is not None:
        del self.asr_model
    except Exception:
      pass
    self.asr_model = None
    self.asr_key = None
    self._release_aux_models()


def _handle_command(runner: PersistentWhisperxRunner, cmd_obj: dict[str, Any]) -> bool:
  cmd = str(cmd_obj.get("cmd") or "").strip().lower()
  if cmd == "shutdown":
    return False
  if cmd == "prewarm":
    response_path = Path(str(cmd_obj.get("response_path") or ""))
    if not response_path:
      return True
    language = _normalize_optional_language(cmd_obj.get("language"))
    align_enabled = bool(cmd_obj.get("align_enabled", False))
    try:
      out = runner.prewarm(language=language, align_enabled=align_enabled)
    except Exception as e:
      out = {
        "ok": False,
        "error": {
          "code": "ASR_PERSISTENT_PREWARM_FAILURE",
          "message": f"Persistent prewarm error: {e!r}",
          "retryable": True,
          "details": {"exc_type": type(e).__name__},
        },
      }
    try:
      _write_json_atomic(response_path, out)
    except Exception:
      pass
    return True
  if cmd != "transcribe":
    return True
  payload_path = Path(str(cmd_obj.get("payload_path") or ""))
  response_path = Path(str(cmd_obj.get("response_path") or ""))
  if not payload_path or not response_path:
    return True
  try:
    envelope = _read_json(payload_path)
    progress_path_raw = str(cmd_obj.get("progress_path") or "").strip()
    progress_path = Path(progress_path_raw) if progress_path_raw else None
    response = runner.transcribe(envelope, progress_path=progress_path)
  except Exception as e:
    request = {}
    try:
      envelope = _read_json(payload_path)
      request = dict(envelope.get("request") or {})
    except Exception:
      request = {}
    response = {
      "schema_version": ASR_SCHEMA_VERSION,
      "request_id": str(request.get("request_id") or ""),
      "ok": False,
      "effective_options": dict(request.get("effective_options") or {}),
      "error": {
        "code": "ASR_PERSISTENT_SERVER_FAILURE",
        "message": f"Persistent server error: {e!r}",
        "retryable": True,
        "details": {"exc_type": type(e).__name__},
      },
      "warnings": [],
    }
  try:
    _write_json_atomic(response_path, response)
  except Exception:
    pass
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description="Persistent WhisperX runner for local ASR requests")
  parser.add_argument("--init-json", required=True)
  ns = parser.parse_args()
  init_obj = _read_json(Path(ns.init_json))
  cfg = dict(init_obj.get("cfg") or {})
  runner = PersistentWhisperxRunner(cfg=cfg)

  for raw in sys.stdin:
    line = str(raw or "").strip()
    if not line:
      continue
    try:
      cmd_obj = json.loads(line)
    except Exception:
      continue
    if not _handle_command(runner, cmd_obj):
      break

  try:
    runner.shutdown()
  except Exception:
    pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
