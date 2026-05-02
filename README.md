# asr-pool

`asr-pool` is a standalone ASR web service that accepts audio jobs, schedules
them over warm WhisperX runner slots, and exposes request status, streaming
completions, and artifact retrieval for client applications.

## What It Does

- accepts audio jobs over a multipart web API
- queues requests as either `interactive` or default non-interactive work
- executes work on warm persistent WhisperX runners
- stores request records and serves generated SRT artifacts
- supports both point-in-time status reads and streaming completion delivery
- exposes pool status and observability endpoints for operators

## Use Cases

For developers building on top of `asr-pool`:

- Client application integration:
  Build a higher-level client around request submission, progress reads, completion streams, and artifact retrieval.

Possible end-user applications:

- Live recording transcription:
  Submit short audio chunks from an ongoing recording and consume streaming completions to keep the transcript up to date while recording continues.

- File-based transcription:
  Submit one file or many files, check progress as requests run, and fetch the generated transcriptions when requests complete.

## Runtime Model

`asr-pool` combines four concerns in one standalone service:

- request intake and validation
- in-process queueing and scheduling
- warm WhisperX runner lifecycle management
- request record, completion feed, and artifact serving

At runtime, the pool keeps a configurable number of runner slots warm and
dispatches queued work onto those slots. The scheduler supports:

- an `interactive` queue and a non-interactive `normal` queue
- fairness for interactive sessions via `routing.fairness_key`
- reserved interactive runner slots that non-interactive work cannot consume
- completion delivery through both polling and streaming

This makes the service useful both for direct clients and for higher-level
consumers such as workers or typed client libraries.

## Configuration

Configuration files are loaded in this order:

1. `config/settings.json`
2. `config/local.json` (optional, overrides)

Primary configuration areas:

- `scheduler.*`
  controls runner slot count, interactive reservation, queue limits, timeouts, and interactive fairness
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
    "runner_slots": 1,
    "interactive_reserved_slots": 0
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

The endpoints are meant to be combined differently per client pattern:

- Live recording clients submit short chunks with `priority: "interactive"`.
  They usually set `routing.fairness_key` to the live session id and request
  `outputs.srt_inline=true`, then keep `GET /asr/v1/completions/stream` open.
  Terminal completion events can carry the SRT text inline at
  `response.result.srt_text`, which avoids a follow-up artifact fetch per chunk.
- File upload or batch clients use the default `priority: "normal"`. They can
  poll `GET /asr/v1/pending-status` or `GET /asr/v1/requests/{request_id}` for
  progress, consume the completion feed as the terminal signal, and then fetch
  the durable SRT from `GET /asr/v1/requests/{request_id}/artifacts/srt`.
- `GET /asr/v1/requests/{request_id}` is useful for one-request inspection,
  recovery, and debugging. It is not the preferred high-throughput event path.
- `GET /asr/v1/pool` and the `/ops` endpoints are operator views for queue,
  slot, and runtime health.

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

`priority: "interactive"` opts a request into latency-sensitive scheduling for
live chunks. Requests without that exact priority use the default `normal` path
for upload and batch work. For interactive fairness, clients may set
`routing.fairness_key`; requests cannot target runner slots directly.

Output flags are explicit. `outputs.srt=true` writes an SRT artifact.
`outputs.srt_inline=true` also embeds SRT text in the terminal response, which
is useful for small interactive chunks. The current persistent runner does not
populate `text` or `segments`, so clients should set those outputs to `false`.

Interactive chunk example:

```bash
curl -sS -X POST http://127.0.0.1:8090/asr/v1/requests \
  -F 'request_json={"schema_version":"asr_v2","request_id":"live_demo_1","priority":"interactive","consumer_id":"live-client","routing":{"fairness_key":"session-1"},"audio":{"format":"wav"},"options":{"language":"nl"},"outputs":{"srt":true,"srt_inline":true,"text":false,"segments":false}}' \
  -F 'audio_file=@/path/to/chunk.wav'
```

Normal upload example:

```bash
curl -sS -X POST http://127.0.0.1:8090/asr/v1/requests \
  -F 'request_json={"schema_version":"asr_v2","request_id":"upload_demo_1","priority":"normal","consumer_id":"upload-worker","audio":{"format":"wav"},"options":{"language":"nl","align_enabled":true},"outputs":{"srt":true,"srt_inline":false,"text":false,"segments":false}}' \
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

## Testing

Run the unit test suite with:

```bash
python3 -m unittest
```

## Acknowledgments

This pool builds on a number of excellent upstream projects:

- FastAPI
- Uvicorn
- WhisperX
- PyTorch

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
