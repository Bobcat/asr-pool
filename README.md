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
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv

ASR_POOL_DIR="$HOME/projects/asr-pool"
mkdir -p "$ASR_POOL_DIR"
git clone https://github.com/Bobcat/asr-pool.git "$ASR_POOL_DIR"
cd "$ASR_POOL_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install whisperx faster-whisper
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 18090
```

Service status endpoint:

- `GET http://127.0.0.1:18090/asr/v1/pool`

Observability endpoints:

- `GET http://127.0.0.1:18090/ops` (live operator page)
- `GET http://127.0.0.1:18090/ops/metrics` (JSON metrics envelope: `service`, `version`, `now_utc`, `window_s`, `health`, `summary`, `details`)

## Systemd Example

```bash
ASR_POOL_DIR="$HOME/projects/asr-pool"

mkdir -p ~/.config/systemd/user ~/.config/asr-pool
cat > ~/.config/systemd/user/asr-pool.service <<EOF
[Unit]
Description=ASR Pool (Uvicorn)

[Service]
Type=simple
WorkingDirectory=$ASR_POOL_DIR
ExecStart=$ASR_POOL_DIR/.venv/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port 18090
Restart=always
RestartSec=2
EnvironmentFile=-%h/.config/asr-pool/asr-pool.env

[Install]
WantedBy=default.target
EOF
cp "$ASR_POOL_DIR/deploy/env/asr-pool.env.example" ~/.config/asr-pool/asr-pool.env
systemctl --user daemon-reload
systemctl --user enable --now asr-pool.service
```

Set `HF_TOKEN` in `~/.config/asr-pool/asr-pool.env` when you need diarization models that require Hugging Face access.

## Configuration

Configuration files are loaded in this order:

1. `config/settings.json`
2. `config/local.json` (optional, overrides)

Default runtime is GPU (`whisperx.device = "cuda"`).

Example local override (`config/local.json`) to run with one slot on CPU-only machines:

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

## Retrieve Results

After completion, fetch results from:

- `GET /asr/v1/requests/{request_id}/artifacts/srt`
