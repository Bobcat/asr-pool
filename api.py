from __future__ import annotations

import asyncio
import json
from email.parser import BytesParser
from email.policy import default as email_policy_default
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from pool_config import get_float, get_str
from pool_helpers import _iso_utc, _safe_token
from pool_service import AsrPoolService


ROOT_PATH = get_str("api.root_path", "")
app = FastAPI(root_path=ROOT_PATH)
ASR_COMPLETIONS_STREAM_HEARTBEAT_S = get_float("completions.stream.sse_heartbeat_s", 10.0, min_value=1.0)
POOL = AsrPoolService()


def _safe_suffix(*, filename: str, audio_format: str) -> str:
    suffix = str(Path(str(filename or "")).suffix or "").strip().lower()
    if not suffix and audio_format:
        fmt = str(audio_format).strip().lower()
        if fmt:
            suffix = f".{fmt}"
    clean = "".join(ch for ch in suffix if ch.isalnum() or ch in {".", "_", "-"}).strip()
    if clean and not clean.startswith("."):
        clean = "." + clean
    clean = clean[:16]
    return clean or ".bin"


def _parse_multipart_submit_payload(*, content_type: str, body: bytes) -> tuple[str, str, bytes]:
    ctype = str(content_type or "").strip().lower()
    if "multipart/form-data" not in ctype:
        raise ValueError("Expected Content-Type multipart/form-data")
    if not body:
        raise ValueError("Empty multipart request body")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=email_policy_default).parsebytes(header + bytes(body))
    if not msg.is_multipart():
        raise ValueError("Malformed multipart request body")
    request_json = None
    audio_filename = ""
    audio_bytes = None
    for part in msg.iter_parts():
        disposition = str(part.get("Content-Disposition") or "").lower()
        if "form-data" not in disposition:
            continue
        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        payload = bytes(part.get_payload(decode=True) or b"")
        if field_name == "request_json":
            request_json = payload.decode("utf-8", errors="replace")
            continue
        if field_name == "audio_file":
            audio_filename = str(part.get_filename() or "").strip()
            audio_bytes = payload
            continue
    if request_json is None:
        raise ValueError("Missing multipart field request_json")
    if audio_bytes is None:
        raise ValueError("Missing multipart field audio_file")
    return request_json, audio_filename, bytes(audio_bytes)


def _error(
    status_code: int,
    *,
    code: str,
    message: str,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "code": str(code),
        "message": str(message),
    }
    if retryable is not None:
        payload["retryable"] = bool(retryable)
    if details:
        payload["details"] = dict(details)
    return JSONResponse(status_code=int(status_code), content=payload)


def _sse_json_event(*, event: str, data: dict[str, Any], event_id: str | None = None) -> bytes:
    lines: list[str] = []
    safe_id = str(event_id or "").strip()
    if safe_id:
        lines.append(f"id: {safe_id}")
    safe_event = str(event or "").strip()
    if safe_event:
        lines.append(f"event: {safe_event}")
    payload = json.dumps(data or {}, ensure_ascii=False, separators=(",", ":"))
    for row in payload.splitlines() or [""]:
        lines.append(f"data: {row}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


@app.on_event("startup")
async def _startup() -> None:
    await POOL.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await POOL.stop()


@app.post("/asr/v1/requests")
async def submit_asr_request(request: Request) -> JSONResponse:
    content_type = str(request.headers.get("content-type") or "").strip()
    try:
        raw_body = await request.body()
    except Exception as e:
        return _error(
            400,
            code="ASR_UPLOAD_READ_FAILED",
            message=f"Failed to read upload body: {type(e).__name__}: {e}",
            retryable=False,
        )
    try:
        request_json, audio_filename, audio_bytes = _parse_multipart_submit_payload(
            content_type=content_type,
            body=bytes(raw_body),
        )
    except ValueError as e:
        return _error(
            400,
            code="ASR_MULTIPART_INVALID",
            message=str(e),
            retryable=False,
        )

    try:
        parsed = json.loads(str(request_json or ""))
    except Exception:
        return _error(
            400,
            code="ASR_REQUEST_JSON_INVALID",
            message="Invalid multipart field request_json (expected JSON object)",
            retryable=False,
        )
    if not isinstance(parsed, dict):
        return _error(
            400,
            code="ASR_REQUEST_JSON_INVALID",
            message="request_json must be a JSON object",
            retryable=False,
        )

    raw_payload = dict(parsed)
    request_id = str(raw_payload.get("request_id") or "").strip()
    if not request_id:
        return _error(
            400,
            code="ASR_REQUEST_ID_REQUIRED",
            message="request_id is required",
            retryable=False,
        )
    audio = dict(raw_payload.get("audio") or {})
    upload_root = (POOL._work_root / "_uploads").resolve()
    upload_dir = (upload_root / _safe_token(request_id)).resolve()
    try:
        upload_dir.relative_to(upload_root)
    except ValueError:
        return _error(
            400,
            code="ASR_UPLOAD_PATH_INVALID",
            message="Invalid upload path for request_id",
            retryable=False,
            details={"request_id": request_id},
        )
    upload_dir.mkdir(parents=True, exist_ok=True)
    dst = (
        upload_dir
        / f"input{_safe_suffix(filename=str(audio_filename or ''), audio_format=str(audio.get('format') or ''))}"
    ).resolve()
    try:
        dst.relative_to(upload_root)
    except ValueError:
        return _error(
            400,
            code="ASR_UPLOAD_PATH_INVALID",
            message="Invalid upload target path",
            retryable=False,
            details={"request_id": request_id},
        )
    bytes_written = int(len(audio_bytes))
    dst.write_bytes(audio_bytes)
    if bytes_written <= 0:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        return _error(
            400,
            code="ASR_EMPTY_INPUT",
            message="Uploaded audio file is empty",
            retryable=False,
            details={"request_id": request_id},
        )
    audio["local_path"] = str(dst)
    audio.pop("blob_ref", None)
    audio.pop("inline_base64", None)
    raw_payload["audio"] = audio

    status_code, body = await POOL.submit(raw_payload)
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/requests/{request_id}")
async def get_asr_request(request_id: str) -> JSONResponse:
    status_code, body = await POOL.get_request(request_id)
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/requests/{request_id}/artifacts/srt")
async def get_asr_request_srt(request_id: str):
    status_code, body = await POOL.get_request(request_id)
    if int(status_code) != 200:
        return JSONResponse(status_code=int(status_code), content=body)
    state = str(body.get("state") or "").strip().lower()
    if state != "completed":
        return _error(
            409,
            code="ASR_REQUEST_NOT_COMPLETED",
            message="SRT artifact is only available for completed requests",
            retryable=False,
            details={"request_id": str(request_id), "state": state},
        )
    response = dict(body.get("response") or {})
    result = dict(response.get("result") or {})
    artifacts = dict(result.get("artifacts") or {})
    srt_path_str = str(artifacts.get("srt_path") or "").strip()
    if not srt_path_str:
        return _error(
            404,
            code="ASR_ARTIFACT_NOT_FOUND",
            message="Completed request has no SRT artifact path",
            retryable=False,
            details={"request_id": str(request_id)},
        )
    srt_path = Path(srt_path_str).resolve()
    try:
        srt_path.relative_to(POOL._work_root)
    except ValueError:
        return _error(
            400,
            code="ASR_ARTIFACT_PATH_INVALID",
            message="SRT artifact path is outside pool work_root",
            retryable=False,
            details={"request_id": str(request_id)},
        )
    if not srt_path.exists() or not srt_path.is_file():
        return _error(
            404,
            code="ASR_ARTIFACT_NOT_FOUND",
            message="SRT artifact file is missing",
            retryable=False,
            details={"request_id": str(request_id)},
        )
    return FileResponse(
        path=str(srt_path),
        media_type="application/x-subrip",
        filename=srt_path.name,
    )


@app.post("/asr/v1/requests/{request_id}/cancel")
async def cancel_asr_request(request_id: str) -> JSONResponse:
    status_code, body = await POOL.cancel(request_id)
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/completions")
async def get_asr_completions(
    consumer_id: str = Query(default=""),
    since_seq: int = Query(default=0),
    limit: int = Query(default=100),
) -> JSONResponse:
    status_code, body = await POOL.completions(
        consumer_id=str(consumer_id),
        since_seq=int(since_seq),
        limit=int(limit),
    )
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/completions/stream")
async def stream_asr_completions(
    consumer_id: str = Query(default=""),
    since_seq: int = Query(default=0),
    limit: int = Query(default=100),
    heartbeat_s: float = Query(default=ASR_COMPLETIONS_STREAM_HEARTBEAT_S),
):
    cid = str(consumer_id or "").strip()
    if not cid:
        return _error(
            400,
            code="ASR_COMPLETIONS_CONSUMER_REQUIRED",
            message="consumer_id is required",
            retryable=False,
        )
    safe_since_seq = max(0, int(since_seq))
    safe_limit = max(1, min(1000, int(limit)))
    safe_heartbeat_s = max(1.0, min(60.0, float(heartbeat_s or ASR_COMPLETIONS_STREAM_HEARTBEAT_S)))

    async def _stream():
        since_local = int(safe_since_seq)
        last_feed_id = ""
        first_meta = True
        try:
            while True:
                status_code, body = await POOL.completions_wait(
                    consumer_id=cid,
                    since_seq=since_local,
                    limit=safe_limit,
                    wait_timeout_s=safe_heartbeat_s,
                )
                if int(status_code) != 200:
                    yield _sse_json_event(
                        event="error",
                        data=dict(body or {}),
                    )
                    break

                feed_id = str(body.get("feed_id") or "")
                next_seq_raw = body.get("next_seq")
                if next_seq_raw is None:
                    next_seq = max(0, int(since_local))
                else:
                    next_seq = max(0, int(next_seq_raw))
                if first_meta or (feed_id and feed_id != last_feed_id):
                    yield _sse_json_event(
                        event="meta",
                        data={
                            "schema": "asr.completions.stream.v1",
                            "consumer_id": str(cid),
                            "feed_id": str(feed_id),
                            "since_seq": int(since_local),
                            "next_seq": int(next_seq),
                        },
                        event_id=str(next_seq),
                    )
                    first_meta = False
                    last_feed_id = str(feed_id)

                emitted_completion = False
                for row in (body.get("events") or []):
                    if not isinstance(row, dict):
                        continue
                    seq = max(0, int(row.get("seq") or 0))
                    yield _sse_json_event(
                        event="completion",
                        data=dict(row),
                        event_id=(str(seq) if seq > 0 else None),
                    )
                    emitted_completion = True
                since_local = int(next_seq)

                if not emitted_completion:
                    yield _sse_json_event(
                        event="heartbeat",
                        data={
                            "feed_id": str(feed_id),
                            "next_seq": int(since_local),
                            "ts_utc": _iso_utc(),
                        },
                        event_id=str(since_local),
                    )
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/asr/v1/pending-status")
async def get_asr_pending_status(
    consumer_id: str = Query(default=""),
    request_id: list[str] = Query(default=[]),
    limit: int = Query(default=200),
) -> JSONResponse:
    status_code, body = await POOL.pending_status(
        consumer_id=str(consumer_id),
        request_ids=[str(v) for v in list(request_id or [])],
        limit=int(limit),
    )
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/pool")
async def get_asr_pool_status(_req: Request) -> JSONResponse:
    body = await POOL.pool_status()
    return JSONResponse(status_code=200, content=body)


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    return _error(500, code="ASR_INTERNAL_ERROR", message=str(exc), retryable=True)
