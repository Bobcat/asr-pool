from __future__ import annotations

import json
import mimetypes
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


class PoolTransportMultipartBuildError(RuntimeError):
  pass


@dataclass(frozen=True)
class PoolTransportConfig:
  base_url: str
  token: str = ""
  http_timeout_s: float = 10.0
  retry_attempts: int = 3
  retry_base_delay_s: float = 0.2
  retry_max_delay_s: float = 2.0
  retry_jitter_s: float = 0.1
  stream_heartbeat_s: float = 10.0

  def normalized(self) -> "PoolTransportConfig":
    base = str(self.base_url or "").strip().rstrip("/")
    if not base:
      base = "http://127.0.0.1:8090"
    timeout_s = max(1.0, float(self.http_timeout_s))
    retry_attempts = max(1, int(self.retry_attempts))
    retry_base = max(0.0, float(self.retry_base_delay_s))
    retry_max = max(0.05, float(self.retry_max_delay_s))
    retry_max = max(retry_base, retry_max)
    retry_jitter = max(0.0, float(self.retry_jitter_s))
    heartbeat_s = max(1.0, float(self.stream_heartbeat_s))
    return PoolTransportConfig(
      base_url=base,
      token=str(self.token or ""),
      http_timeout_s=timeout_s,
      retry_attempts=retry_attempts,
      retry_base_delay_s=retry_base,
      retry_max_delay_s=retry_max,
      retry_jitter_s=retry_jitter,
      stream_heartbeat_s=heartbeat_s,
    )


def _json_or_empty(raw: bytes) -> dict[str, Any]:
  if not raw:
    return {}
  try:
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
  except Exception:
    return {}
  return dict(parsed) if isinstance(parsed, dict) else {}


def _http_request_once(
  *,
  method: str,
  url: str,
  token: str,
  timeout_s: float,
  body_bytes: bytes | None = None,
  content_type: str | None = None,
  accept: str | None = None,
) -> tuple[int, dict[str, Any]]:
  req = urlrequest.Request(
    url,
    data=(bytes(body_bytes) if body_bytes is not None else None),
    method=str(method).upper(),
  )
  if content_type:
    req.add_header("Content-Type", str(content_type))
  if accept:
    req.add_header("Accept", str(accept))
  if token:
    req.add_header("X-ASR-Token", token)
  try:
    with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
      return int(getattr(resp, "status", 200) or 200), _json_or_empty(resp.read())
  except urlerror.HTTPError as e:
    return int(getattr(e, "code", 500) or 500), _json_or_empty(e.read())


def _retryable_http_status(status_code: int) -> bool:
  code = int(status_code)
  return code == 429 or code >= 500


def _backoff_sleep_s(*, retry_index: int, base_s: float, max_s: float, jitter_s: float) -> float:
  idx = max(0, int(retry_index))
  expo = float(base_s) * (2 ** idx)
  bounded = min(float(max_s), max(0.0, float(expo)))
  if float(jitter_s) > 0.0:
    bounded += random.uniform(0.0, float(jitter_s))
  return max(0.0, float(bounded))


def _http_request_with_retry(
  *,
  config: PoolTransportConfig,
  method: str,
  url: str,
  body_bytes: bytes | None = None,
  content_type: str | None = None,
  accept: str | None = None,
) -> tuple[int, dict[str, Any], int]:
  cfg = config.normalized()
  max_attempts = max(1, int(cfg.retry_attempts))
  last_exc: Exception | None = None
  for attempt in range(1, max_attempts + 1):
    try:
      status_code, body = _http_request_once(
        method=method,
        url=url,
        token=cfg.token,
        timeout_s=cfg.http_timeout_s,
        body_bytes=body_bytes,
        content_type=content_type,
        accept=accept,
      )
    except Exception as e:
      last_exc = e
      if attempt >= max_attempts:
        raise
      sleep_s = _backoff_sleep_s(
        retry_index=(attempt - 1),
        base_s=cfg.retry_base_delay_s,
        max_s=cfg.retry_max_delay_s,
        jitter_s=cfg.retry_jitter_s,
      )
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      continue

    if _retryable_http_status(status_code) and attempt < max_attempts:
      sleep_s = _backoff_sleep_s(
        retry_index=(attempt - 1),
        base_s=cfg.retry_base_delay_s,
        max_s=cfg.retry_max_delay_s,
        jitter_s=cfg.retry_jitter_s,
      )
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      continue
    return int(status_code), dict(body or {}), int(attempt)

  if last_exc is not None:
    raise last_exc
  return 500, {}, int(max_attempts)


def _multipart_content_type_for_path(path: Path) -> str:
  guessed, _enc = mimetypes.guess_type(str(path.name))
  return str(guessed or "application/octet-stream")


def _build_multipart_submit_body(
  *,
  request_payload: dict[str, Any],
  audio_path: Path,
) -> tuple[bytes, str]:
  boundary = f"----asr-{uuid.uuid4().hex}"
  request_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
  file_bytes = audio_path.read_bytes()
  filename = str(audio_path.name or "audio.bin").replace("\"", "_")
  file_content_type = _multipart_content_type_for_path(audio_path)
  rows: list[bytes] = []
  rows.append(f"--{boundary}\r\n".encode("ascii"))
  rows.append(b"Content-Disposition: form-data; name=\"request_json\"\r\n")
  rows.append(b"Content-Type: application/json; charset=utf-8\r\n\r\n")
  rows.append(request_bytes)
  rows.append(b"\r\n")
  rows.append(f"--{boundary}\r\n".encode("ascii"))
  rows.append(f"Content-Disposition: form-data; name=\"audio_file\"; filename=\"{filename}\"\r\n".encode("utf-8"))
  rows.append(f"Content-Type: {file_content_type}\r\n\r\n".encode("ascii"))
  rows.append(file_bytes)
  rows.append(b"\r\n")
  rows.append(f"--{boundary}--\r\n".encode("ascii"))
  return b"".join(rows), f"multipart/form-data; boundary={boundary}"


def submit_multipart_request(
  *,
  config: PoolTransportConfig,
  request_payload: dict[str, Any],
  audio_path: Path,
) -> tuple[int, dict[str, Any], int]:
  cfg = config.normalized()
  submit_url = urlparse.urljoin(cfg.base_url + "/", "asr/v1/requests")
  try:
    body_bytes, content_type = _build_multipart_submit_body(
      request_payload=request_payload,
      audio_path=audio_path,
    )
  except Exception as e:
    raise PoolTransportMultipartBuildError(f"{type(e).__name__}: {e}") from e
  return _http_request_with_retry(
    config=cfg,
    method="POST",
    url=submit_url,
    body_bytes=body_bytes,
    content_type=content_type,
  )


def fetch_pending_status(
  *,
  config: PoolTransportConfig,
  consumer_id: str,
  request_ids: list[str],
  limit: int = 200,
) -> list[dict[str, Any]]:
  cfg = config.normalized()
  clean_ids: list[str] = []
  seen: set[str] = set()
  for raw in list(request_ids or []):
    rid = str(raw or "").strip()
    if not rid or rid in seen:
      continue
    seen.add(rid)
    clean_ids.append(rid)
    if len(clean_ids) >= int(max(1, min(1000, int(limit)))):
      break

  query = urlparse.urlencode(
    {
      "consumer_id": str(consumer_id or ""),
      "limit": int(max(1, min(1000, int(limit)))),
      "request_id": clean_ids,
    },
    doseq=True,
  )
  url = urlparse.urljoin(cfg.base_url + "/", f"asr/v1/pending-status?{query}")
  try:
    status_code, body, _attempts_used = _http_request_with_retry(
      config=cfg,
      method="GET",
      url=url,
      content_type="application/json",
    )
  except Exception:
    return []
  if int(status_code) != 200:
    return []
  rows = dict(body or {}).get("rows")
  if not isinstance(rows, list):
    return []
  return [row for row in rows if isinstance(row, dict)]


def download_request_srt_to_path(
  *,
  config: PoolTransportConfig,
  request_id: str,
  dst_path: Path,
  allow_empty: bool = False,
) -> Path:
  cfg = config.normalized()
  rid = str(request_id or "").strip()
  if not rid:
    raise RuntimeError("Missing request_id for remote SRT fetch")
  safe_rid = urlparse.quote(rid, safe="")
  url = urlparse.urljoin(cfg.base_url + "/", f"asr/v1/requests/{safe_rid}/artifacts/srt")
  req = urlrequest.Request(url, method="GET")
  if cfg.token:
    req.add_header("X-ASR-Token", cfg.token)
  try:
    with urlrequest.urlopen(req, timeout=float(max(5.0, cfg.http_timeout_s))) as resp:
      status_code = int(getattr(resp, "status", 200) or 200)
      data = bytes(resp.read() or b"")
      if status_code != 200:
        raise RuntimeError(f"ASR_REMOTE_SRT_FETCH_FAILED: Failed to fetch remote SRT (http={status_code})")
  except urlerror.HTTPError as e:
    status_code = int(getattr(e, "code", 500) or 500)
    body = _json_or_empty(e.read())
    code = str(body.get("code") or "ASR_REMOTE_SRT_FETCH_FAILED")
    msg = str(body.get("message") or f"Failed to fetch remote SRT (http={status_code})")
    raise RuntimeError(f"{code}: {msg}") from e
  except RuntimeError:
    raise
  except Exception as e:
    raise RuntimeError(f"ASR_REMOTE_ARTIFACT_FETCH_IO_FAILURE: {type(e).__name__}: {e}") from e
  if not data and not bool(allow_empty):
    raise RuntimeError("Remote SRT fetch returned empty payload")
  dst_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
  tmp.write_bytes(data)
  tmp.replace(dst_path)
  return dst_path.resolve()


def _parse_sse_event(*, event_name: str, data_lines: list[str]) -> tuple[str, dict[str, Any]]:
  raw = "\n".join(list(data_lines or [])).strip()
  if not raw:
    return str(event_name or "message").strip().lower(), {}
  try:
    parsed = json.loads(raw)
  except Exception:
    return str(event_name or "message").strip().lower(), {"raw": raw}
  return str(event_name or "message").strip().lower(), (dict(parsed) if isinstance(parsed, dict) else {"value": parsed})


def stream_completions_forever(
  *,
  config: PoolTransportConfig,
  consumer_id: str,
  start_since_seq: int,
  stop_event: threading.Event,
  on_event: Callable[[str, dict[str, Any]], None],
) -> None:
  cfg = config.normalized()
  cid = str(consumer_id or "").strip()
  if not cid:
    on_event(
      "stream_error",
      {
        "code": "ASR_COMPLETIONS_STREAM_CONSUMER_REQUIRED",
        "message": "consumer_id is required for stream reader",
      },
    )
    return

  timeout_s = max(30.0, (float(cfg.stream_heartbeat_s) * 3.0))
  since_seq = max(0, int(start_since_seq))
  last_feed_id = ""
  retry_index = 0

  while not stop_event.is_set():
    query = urlparse.urlencode(
      {
        "consumer_id": cid,
        "since_seq": max(0, int(since_seq)),
        "limit": 200,
        "heartbeat_s": float(cfg.stream_heartbeat_s),
      }
    )
    stream_url = urlparse.urljoin(cfg.base_url + "/", f"asr/v1/completions/stream?{query}")
    req = urlrequest.Request(stream_url, method="GET")
    req.add_header("Accept", "text/event-stream")
    if cfg.token:
      req.add_header("X-ASR-Token", cfg.token)
    try:
      with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
        status_code = int(getattr(resp, "status", 200) or 200)
        if status_code != 200:
          on_event(
            "stream_error",
            {
              "code": "ASR_COMPLETIONS_STREAM_HTTP_ERROR",
              "status_code": int(status_code),
              "url": stream_url,
            },
          )
          raise RuntimeError(f"completions stream http={status_code}")
        retry_index = 0
        event_name = "message"
        data_lines: list[str] = []
        while not stop_event.is_set():
          raw_line = resp.readline()
          if not raw_line:
            break
          try:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
          except Exception:
            line = str(raw_line).rstrip("\r\n")
          if line == "":
            kind, payload = _parse_sse_event(event_name=event_name, data_lines=data_lines)
            event_name = "message"
            data_lines = []
            if kind == "meta":
              feed_id = str(payload.get("feed_id") or "").strip()
              next_seq_raw = payload.get("next_seq")
              next_seq = max(0, int(next_seq_raw or 0))
              feed_changed = bool(last_feed_id and feed_id and feed_id != last_feed_id)
              if feed_changed:
                on_event(
                  "feed_reset",
                  {
                    "old_feed_id": str(last_feed_id),
                    "new_feed_id": str(feed_id),
                  },
                )
              if feed_id:
                last_feed_id = feed_id
              if feed_changed:
                since_seq = 0
                break
              elif next_seq_raw is not None:
                since_seq = max(since_seq, next_seq)
            elif kind == "completion":
              seq = max(0, int(payload.get("seq") or 0))
              if seq > 0:
                since_seq = max(since_seq, seq + 1)
              on_event("completion", dict(payload))
            elif kind == "heartbeat":
              next_seq = max(0, int(payload.get("next_seq") or since_seq))
              feed_id = str(payload.get("feed_id") or "").strip()
              since_seq = max(since_seq, next_seq)
              if feed_id:
                last_feed_id = feed_id
            elif kind == "error":
              on_event(
                "stream_error",
                {
                  "code": str(payload.get("code") or "ASR_COMPLETIONS_STREAM_ERROR_EVENT"),
                  "message": str(payload.get("message") or "stream error event"),
                  "payload": dict(payload),
                },
              )
            continue
          if line.startswith(":"):
            continue
          if line.startswith("event:"):
            event_name = str(line[6:]).strip() or "message"
            continue
          if line.startswith("data:"):
            data_lines.append(str(line[5:]).lstrip())
            continue
    except Exception as e:
      if stop_event.is_set():
        break
      on_event(
        "stream_error",
        {
          "code": "ASR_COMPLETIONS_STREAM_IO_FAILURE",
          "message": f"{type(e).__name__}: {e}",
          "retryable": True,
        },
      )
      sleep_s = _backoff_sleep_s(
        retry_index=retry_index,
        base_s=cfg.retry_base_delay_s,
        max_s=cfg.retry_max_delay_s,
        jitter_s=cfg.retry_jitter_s,
      )
      retry_index += 1
      if sleep_s > 0.0 and not stop_event.is_set():
        stop_event.wait(timeout=float(sleep_s))
