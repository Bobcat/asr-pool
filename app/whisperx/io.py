from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  os.replace(tmp, path)


def _now_iso() -> str:
  from datetime import datetime, timezone
  return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _normalize_progress_timings(timings: dict[str, Any] | None) -> dict[str, float]:
  out: dict[str, float] = {}
  for raw_key, raw_value in dict(timings or {}).items():
    key = str(raw_key or "").strip()
    if not key:
      continue
    try:
      sec = max(0.0, float(raw_value))
    except Exception:
      continue
    out[key] = round(sec, 6)
  return out


def _write_progress(progress_path: Path | None, *, stage: str, timings: dict[str, Any] | None = None) -> None:
  if progress_path is None:
    return
  try:
    payload = {
      "stage": str(stage or "").strip().lower(),
      "ts_utc": _now_iso(),
    }
    safe_timings = _normalize_progress_timings(timings)
    if safe_timings:
      payload["timings"] = safe_timings
    _write_json_atomic(progress_path, payload)
  except Exception:
    pass
