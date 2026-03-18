# asr-pool

`asr-pool` is an HTTP ASR service that accepts audio jobs, schedules them over warm WhisperX runner slots, and exposes completion + artifact endpoints for client applications.

## What It Does

- Accepts ASR jobs over HTTP (`multipart/form-data`)
- Queues jobs by priority (`interactive`, `normal`, `background`)
- Executes jobs on warm persistent WhisperX runners
- Stores request records and serves generated SRT artifacts
- Supports both request/response and streaming delivery for completion updates

## Quick Start

```bash
git clone git@github.com:Bobcat/asr-pool.git <path-to-asr-pool-dir>
cd <path-to-asr-pool-dir>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 18090
```

Service status endpoint:

- `GET http://127.0.0.1:18090/asr/v1/pool`

## Configuration

Configuration files are loaded in this order:

1. `config/settings.json`
2. `config/local.json` (optional, gitignored)

Common settings:

- `paths.work_root`: local storage root for job inputs/results
- `scheduler.runner_slots`: number of warm runner slots (default `2`)
- `scheduler.queue_limits.*`: queue caps by priority
- `scheduler.request_timeouts_s.*`: timeout defaults by priority
- `whisperx.*`: WhisperX model/runtime options

Example local override (`config/local.json`) to run with one slot:

```json
{
  "scheduler": {
    "runner_slots": 1
  }
}
```

## API Overview

- `POST /asr/v1/requests`
- `GET /asr/v1/requests/{request_id}`
- `GET /asr/v1/requests/{request_id}/artifacts/srt`
- `POST /asr/v1/requests/{request_id}/cancel`
- `GET /asr/v1/completions`
- `GET /asr/v1/completions/stream`
- `GET /asr/v1/pending-status`
- `GET /asr/v1/pool`

## Submit Contract (Multipart)

`POST /asr/v1/requests` expects:

- field `request_json` (JSON string)
- field `audio_file` (binary file)

Required keys in `request_json`:

- `schema_version` (`"asr_v2"`)
- `request_id`

Minimal example:

```bash
curl -sS -X POST http://127.0.0.1:18090/asr/v1/requests \
  -F 'request_json={"schema_version":"asr_v2","request_id":"job_demo_1","priority":"interactive","consumer_id":"client-1","audio":{"format":"wav"},"options":{"language":"nl"},"outputs":{"srt":true}}' \
  -F 'audio_file=@/path/to/audio.wav'
```

After completion, fetch subtitles from:

- `GET /asr/v1/requests/{request_id}/artifacts/srt`

## Runtime Notes

- Uploaded audio and generated artifacts are written under `paths.work_root`.
- Completion streaming uses Server-Sent Events on `/asr/v1/completions/stream`.
