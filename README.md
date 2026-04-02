# asr-pool

`asr-pool` is a standalone ASR web service that accepts audio jobs, schedules
them over warm WhisperX runner slots, and exposes request status, streaming
completions, and artifact retrieval for client applications.

## What It Does

- accepts audio jobs over a multipart web API
- queues requests by priority: `interactive`, `normal`, `background`
- executes work on warm persistent WhisperX runners
- stores request records and serves generated SRT artifacts
- supports both point-in-time status reads and streaming completion delivery
- exposes pool status and observability endpoints for operators

## Runtime Model

`asr-pool` combines four concerns in one standalone service:

- request intake and validation
- in-process queueing and scheduling
- warm WhisperX runner lifecycle management
- request record, completion feed, and artifact serving

At runtime, the pool keeps a configurable number of runner slots warm and
dispatches queued work onto those slots. The scheduler supports:

- priority queues for `interactive`, `normal`, and `background`
- fairness for interactive sessions via `routing.fairness_key`
- optional slot targeting via `routing.slot_affinity`
- completion delivery through both polling and streaming

This makes the service useful both for direct clients and for higher-level
consumers such as workers or typed client libraries.

## Configuration

Configuration files are loaded in this order:

1. `config/settings.json`
2. `config/local.json` (optional, overrides)

Primary configuration areas:

- `scheduler.*`
  controls runner slot count, queue limits, timeouts, and interactive fairness
- `lifecycle.warm_start.*`
  controls prewarming of WhisperX runner slots
- `lifecycle.watchdog.*`
  controls health checking and recovery of warm runners
- `paths.work_root`
  controls where uploads, request records, and generated artifacts are stored
- `completions.*`
  controls completion feed retention and SSE heartbeat timing
- `whisperx.*`
  controls model, device, compute type, chunking, batching, and runner venv

Default runtime is GPU-oriented:

- `whisperx.device = "cuda"`
- `whisperx.model = "large-v3"`

Example local override for a smaller CPU-only setup:

```json
{
  "scheduler": {
    "runner_slots": 1
  },
  "whisperx": {
    "device": "cpu"
  }
}
```

## API Overview

Request lifecycle endpoints:

- `POST /asr/v1/requests`
- `GET /asr/v1/requests/{request_id}`
- `POST /asr/v1/requests/{request_id}/cancel`
- `GET /asr/v1/requests/{request_id}/artifacts/srt`

Completion and progress endpoints:

- `GET /asr/v1/completions`
- `GET /asr/v1/completions/stream`
- `GET /asr/v1/pending-status`
- `GET /asr/v1/pool`

Observability endpoints:

- `GET /ops`
- `GET /ops/metrics`

The API offers multiple delivery models on purpose:

- direct request lookup for single-request state
- pending-status snapshots for progress-oriented clients
- completion feeds for terminal event delivery
- artifact fetch for generated outputs such as SRT

## Submit Contract

`POST /asr/v1/requests` expects multipart form data with:

- field `request_json` containing a JSON object
- field `audio_file` containing the binary audio payload

Required keys in `request_json`:

- `schema_version` with value `"asr_v2"`
- `request_id`

Common request fields include:

- `consumer_id`
- `priority`
- `audio`
- `options`
- `outputs`
- `routing`

Minimal example:

```bash
curl -sS -X POST http://127.0.0.1:8090/asr/v1/requests \
  -F 'request_json={"schema_version":"asr_v2","request_id":"job_demo_1","priority":"interactive","consumer_id":"client-1","audio":{"format":"wav"},"options":{"language":"nl"},"outputs":{"srt":true}}' \
  -F 'audio_file=@/path/to/audio.wav'
```

## Results And Completions

After submission, clients typically use one or more of these read paths:

- `GET /asr/v1/requests/{request_id}`
  returns the current request record for one request id
- `GET /asr/v1/pending-status`
  returns point-in-time status rows for a known set of pending request ids
- `GET /asr/v1/completions`
  returns completion events from the pool feed
- `GET /asr/v1/completions/stream`
  streams completion events as they happen
- `GET /asr/v1/requests/{request_id}/artifacts/srt`
  returns the generated SRT artifact after completion

This split lets clients choose the right integration style:

- polling for UI progress
- streaming for terminal wakeups
- artifact fetch for final outputs
