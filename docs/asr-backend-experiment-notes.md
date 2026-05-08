# ASR Backend Experiment Notes

## Purpose

This note documents the practical differences between the current WhisperX path
and the experimental `faster_whisper_direct` path in `asr-pool`.

The goal is to make backend behavior understandable enough to run targeted
experiments, and to define the code changes needed to expose faster-whisper
arguments and results without splitting the ASR request lifecycle.

This is not a workflow policy document. It describes backend behavior,
available knobs, result shapes, and implementation work.

## Current Architecture

`asr-pool` exposes one request lifecycle:

- submit audio with `POST /asr/v1/requests`
- queue by priority
- execute on a warm persistent runner slot
- return terminal status through request lookup and completions
- optionally produce artifacts such as SRT

Backend selection currently happens inside the persistent runner. The request
option `options.asr_backend` can select:

- `whisperx`
- `faster_whisper_direct`

If the request does not provide a backend override, the runner uses the global
default `asr.backend`.

## Shared Runtime Facts

Both backend paths currently use the same warm runner process and the same
loaded WhisperX ASR model object.

The model is loaded language-agnostically:

```python
whisperx.load_model(..., language=None, ...)
```

Per-request language hints are applied at transcribe time.

The direct faster-whisper path does not require a dedicated slot at startup. It
uses the underlying faster-whisper/CTranslate2 model inside the loaded WhisperX
pipeline object:

```python
runner.asr_model.model.transcribe(...)
```

This means a single warm slot can run one request through WhisperX and another
request through direct faster-whisper, as long as the loaded model config is
unchanged.

## Pipeline Comparison

| Area | WhisperX path | `faster_whisper_direct` path |
| --- | --- | --- |
| Model loading | `whisperx.load_model(...)` | Same loaded WhisperX object |
| Native call | `runner.asr_model.transcribe(...)` | `runner.asr_model.model.transcribe(...)` |
| Preprocessing | WhisperX VAD, chunk merge, feature extraction, batching | faster-whisper native preprocessing inside `transcribe(...)` |
| Chunk parameter | `chunk_size` controls WhisperX VAD merge chunks | `chunk_length` controls faster-whisper feature/window length |
| VAD | WhisperX pipeline VAD; configured through `vad_options` and transcribe `chunk_size` | faster-whisper `vad_filter` and `vad_parameters` |
| Language | `language=None` detects language; language code fixes it | Same: `language=None` detects, language code fixes it |
| Alignment | WhisperX can run forced alignment after transcription | Not part of raw FW transcribe; can still be followed by WhisperX align if enabled |
| Diarization | WhisperX can run diarization after alignment | Not part of raw FW transcribe; can still be followed by WhisperX diarization if enabled |
| Segment metadata | WhisperX normalizes to a smaller segment dict | Raw FW segment objects contain more decode metadata |
| Best current fit | High-quality longer transcription, alignment, SRT | Low-latency decode experiments and richer raw decode metadata |

WhisperX uses faster-whisper as a lower-level engine, but it does not simply
call `WhisperModel.transcribe(..., chunk_length=...)`. It performs its own VAD
and chunking, then uses lower-level model generation.

## Request Contract Direction

Keep one ASR request type. The request still means "transcribe this audio".
Backend differences are expressed through options and result metadata, not
through separate lifecycle endpoints.

The target contract keeps common options in `options` and adds backend-specific
experiment knobs as explicit option keys.

Do not infer backend behavior from `priority`. `priority="interactive"` means
latency-sensitive scheduling intent; it does not imply a specific backend or
VAD policy.

## Common Request Options

| Option | Applies to | Current support | Target support | Description |
| --- | --- | --- | --- | --- |
| `asr_backend` | All | Yes | Yes | Request-level backend override. Valid values: `whisperx`, `faster_whisper_direct`. |
| `language` | All | Yes | Yes | Language code such as `nl` or `en`; `null`, empty string, and `auto` mean autodetect. |
| `initial_prompt` | All | Yes | Yes | Optional prompt/hint text for decoding. |
| `beam_size` | All | Yes | Yes | Beam search size. |
| `align_enabled` | WhisperX postprocess | Yes | Yes | Enables WhisperX alignment after transcription. |
| `diarize_enabled` | WhisperX postprocess | Yes | Yes | Enables diarization after transcription. |
| `speaker_mode` | WhisperX postprocess | Yes | Yes | `none`, `auto`, or `fixed`. |
| `min_speakers` | WhisperX postprocess | Yes | Yes | Minimum speaker count when fixed diarization is used. |
| `max_speakers` | WhisperX postprocess | Yes | Yes | Maximum speaker count when fixed diarization is used. |
| `chunk_size` | WhisperX | Yes | Yes | WhisperX VAD merge chunk size in seconds. |
| `chunk_length` | faster-whisper | No | Yes | faster-whisper feature/window chunk length in seconds. |
| `vad_filter` | faster-whisper | No | Yes | Enables or disables faster-whisper native VAD. |
| `vad_parameters` | faster-whisper | No | Yes | Dict of faster-whisper VAD parameters. |
| `word_timestamps` | faster-whisper | No | Yes | Enables FW word timestamp extraction. Default false. |
| `max_new_tokens` | faster-whisper | No | Yes | Limits generated tokens per chunk. |
| `hotwords` | faster-whisper | No | Yes | Hint words or phrases. |

## WhisperX Arguments

These are the effective WhisperX arguments currently relevant to `asr-pool`.

| Argument | Source | Description |
| --- | --- | --- |
| `model` | config `whisperx.model` | Whisper model id, for example `large-v3`. |
| `device` | config `whisperx.device` | Runtime device, for example `cuda` or `cpu`. |
| `compute_type` | config `whisperx.compute_type` | CTranslate2 compute type. |
| `batch_size` | config `whisperx.batch_size` | Batch size used by WhisperX transcribe. |
| `chunk_size` | config/request | WhisperX VAD merge chunk size in seconds. |
| `language` | request | Language code or autodetect when null. |
| `beam_size` | config/request | Beam search size. |
| `initial_prompt` | request | Prompt/hint text when supported by the WhisperX ASR model options object. |
| `align_enabled` | request | Runs WhisperX alignment after transcription. |
| `diarize_enabled` | request | Runs diarization after alignment when speaker mode is not `none`. |
| `speaker_mode` | request | Controls diarization speaker handling. |
| `min_speakers` | request | Minimum speakers for fixed diarization. |
| `max_speakers` | request | Maximum speakers for fixed diarization. |
| `align_model` | config `whisperx.align_model` | Optional align model override. |
| `diarize_model` | config `whisperx.diarize_model` | Optional diarization model override. |

WhisperX native `FasterWhisperPipeline.transcribe(...)` also accepts internal
parameters such as `num_workers`, `task`, `print_progress`,
`combined_progress`, and `verbose`. `asr-pool` does not currently expose those
as request options.

## faster-whisper `WhisperModel.transcribe(...)` Arguments

The direct path should target these upstream faster-whisper arguments.

| Argument | Default | Description |
| --- | --- | --- |
| `audio` | required | Audio path, binary IO, or numpy waveform. `asr-pool` passes a loaded audio array. |
| `language` | `None` | Language code. `None` enables language detection. |
| `task` | `transcribe` | `transcribe` or `translate`. |
| `log_progress` | `False` | Logs progress. Not useful for normal pool operation. |
| `beam_size` | `5` | Beam size for decoding. |
| `best_of` | `5` | Candidate count when sampling with non-zero temperature. |
| `patience` | `1` | Beam search patience factor. |
| `length_penalty` | `1` | Length penalty. |
| `repetition_penalty` | `1` | Penalty for repeated tokens. |
| `no_repeat_ngram_size` | `0` | Prevents repeated n-grams when greater than zero. |
| `temperature` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Temperature fallback schedule. |
| `compression_ratio_threshold` | `2.4` | Failure threshold for compression ratio. |
| `log_prob_threshold` | `-1.0` | Failure threshold for average log probability. |
| `no_speech_threshold` | `0.6` | Silence threshold used with log probability. |
| `condition_on_previous_text` | `True` | Reuses previous text as prompt for later windows. Direct low-latency calls often set this false. |
| `prompt_reset_on_temperature` | `0.5` | Resets prompt above this temperature when previous text conditioning is enabled. |
| `initial_prompt` | `None` | Prompt for the first window. |
| `prefix` | `None` | Prefix for the first window. |
| `suppress_blank` | `True` | Suppresses blank outputs at decode start. |
| `suppress_tokens` | `[-1]` | Tokens to suppress; `-1` means default non-speech tokens. |
| `without_timestamps` | `False` | Samples text without timestamp tokens when true. |
| `max_initial_timestamp` | `1.0` | Maximum initial timestamp. |
| `word_timestamps` | `False` | Extracts word timestamps and word probabilities. |
| `prepend_punctuations` | upstream default punctuation set | Punctuation merged with following word for word timestamps. |
| `append_punctuations` | upstream default punctuation set | Punctuation merged with previous word for word timestamps. |
| `multilingual` | `False` | Performs language detection on every segment. |
| `vad_filter` | `False` | Enables Silero VAD filtering inside faster-whisper. |
| `vad_parameters` | `None` | Dict or `VadOptions` for faster-whisper VAD. |
| `max_new_tokens` | `None` | Maximum new tokens per chunk. |
| `chunk_length` | `None` | Audio feature/window chunk length. |
| `clip_timestamps` | `"0"` | Clip ranges to process. VAD is ignored when clip timestamps are used. |
| `hallucination_silence_threshold` | `None` | Skips silence around possible hallucinations when word timestamps are enabled. |
| `hotwords` | `None` | Hotword/hint phrases. |
| `language_detection_threshold` | `0.5` | Probability threshold for language detection. |
| `language_detection_segments` | `1` | Number of segments used for language detection. |

### faster-whisper VAD Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `threshold` | `0.5` | Speech probability threshold. |
| `neg_threshold` | `None` | Negative threshold; faster-whisper derives one when omitted. |
| `min_speech_duration_ms` | `0` | Minimum speech duration to keep. |
| `max_speech_duration_s` | `inf` | Maximum speech duration before splitting. |
| `min_silence_duration_ms` | `2000` | Minimum silence duration used for splitting. |
| `speech_pad_ms` | `400` | Padding around detected speech. |
| `min_silence_at_max_speech` | `98` | Silence rule near max speech duration. |
| `use_max_poss_sil_at_max_speech` | `True` | Whether to use the longest possible silence near max speech duration. |

## Current Direct FW Arguments

The direct-FW path forwards a focused subset to `fw_model.transcribe(...)`:

| Argument | Handling |
| --- | --- |
| `condition_on_previous_text` | `False` |
| `vad_filter` | Request option, defaulting to `True` when absent. |
| `beam_size` | config/request beam size |
| `language` | request language when not null |
| `initial_prompt` | request prompt when not null |
| `chunk_length` | request `chunk_length` when set |
| direct-FW experiment args | request option when set |

`chunk_size` is WhisperX-only and is not sent to faster-whisper.

## Common Target Result Fields

The response should expose a common JSON surface regardless of backend.

| Field | Description |
| --- | --- |
| `result.text` | Combined transcript text when `outputs.text=true`. |
| `result.segments` | JSON transcript segments when `outputs.segments=true`. |
| `result.language.code` | Detected or requested language code when known. |
| `result.language.probability` | Language probability when the backend provides it. |
| `result.artifacts.srt_path` | SRT artifact path when SRT was requested and produced. |
| `result.srt_text` | Inline SRT text when `outputs.srt_inline=true`. |
| `result.backend_metadata` | Backend-specific metadata selected for normal responses. |

`text` and `segments` are inline JSON outputs. `srt` and `srt_inline` remain
SRT artifact/text outputs.

## WhisperX Results

| Field | Exposure | Description |
| --- | --- | --- |
| `segments` | Return when `outputs.segments=true` | WhisperX normalized or aligned segment dicts. |
| `text` | Return when `outputs.text=true` | Combined segment text. |
| `language` | Return under `result.language.code` when known | Language detected or requested by WhisperX. |
| `artifacts.srt_path` | Return when `outputs.srt=true` or `outputs.srt_inline=true` | Path to generated SRT artifact. |
| `srt_text` | Return when `outputs.srt_inline=true` | Inline SRT content. |
| `audio_processed_ms` | Return when derivable | Duration of processed audio. |
| `runtime` | Return separately | Runner/backend metadata. |
| `timings` | Return | Timing measurements. |
| `warnings` | Return | Backend/runtime warnings. |

WhisperX can enrich segments after transcription through alignment and
diarization. Those enriched fields should be preserved when JSON segments are
requested.

## faster-whisper Raw Results

Raw faster-whisper returns:

```python
segments, info = fw_model.transcribe(...)
```

### Raw Segment Fields

| Field | Default exposure | Description |
| --- | --- | --- |
| `id` | No | Segment index from faster-whisper. |
| `seek` | No | Seek position. |
| `start` | Yes | Segment start time in seconds. |
| `end` | Yes | Segment end time in seconds. |
| `text` | Yes | Segment text. |
| `tokens` | No | Token ids. Not exposed in first implementation. |
| `temperature` | Yes | Temperature used for this segment. |
| `avg_logprob` | Yes | Average log probability. |
| `compression_ratio` | Yes | Compression ratio heuristic. |
| `no_speech_prob` | Yes | No-speech probability. |
| `words` | Opt-in | Word timestamps when `word_timestamps=true`. |

### Raw Transcription Info Fields

| Field | Default exposure | Description |
| --- | --- | --- |
| `language` | Yes | Detected or requested language. |
| `language_probability` | Yes | Probability for selected language. |
| `duration` | Yes | Input audio duration. |
| `duration_after_vad` | Yes | Audio duration after FW VAD filtering. |
| `all_language_probs` | No | Full language probability list. Not exposed in first implementation. |
| `transcription_options` | No | Full decode options. Not exposed by default. |
| `vad_options` | No | Full VAD options. Not exposed by default. |

### Default FW Metadata Decision

Default direct-FW metadata is intentionally light:

- per segment: `text`, `start`, `end`, `avg_logprob`,
  `compression_ratio`, `no_speech_prob`, `temperature`
- global: `language`, `language_probability`, `duration`,
  `duration_after_vad`
- `words` only when `word_timestamps=true`
- no raw `tokens`
- no `all_language_probs`
- no full options dump

## Output Selection

The runner should fill `outputs.text=true` and `outputs.segments=true` as
inline JSON outputs.

| Output | Target meaning |
| --- | --- |
| `text` | Include combined transcript text in `result.text`. |
| `segments` | Include JSON segments in `result.segments`. |
| `srt` | Produce SRT artifact and return `result.artifacts.srt_path`. |
| `srt_inline` | Include SRT text in `result.srt_text`. |

SRT should only be produced when `outputs.srt=true` or
`outputs.srt_inline=true`.

## Required Changes

### `asr-pool`

1. Rename global backend default config from `whisperx.low_latency.backend` to
   `asr.backend`.
2. Keep `options.asr_backend` as the request-level backend override.
3. Normalize `language=null`, empty string, and `language="auto"` to internal
   `language=None`.
4. Extend option normalization for direct-FW experiment arguments:
   `vad_filter`, `vad_parameters`, `word_timestamps`, `chunk_length`,
   `max_new_tokens`, `hotwords`, language detection args, and threshold args.
5. Replace the current direct-FW `chunk_size` call argument with
   `chunk_length`.
6. Stop rejecting `outputs.text` and `outputs.segments`.
7. Populate common JSON `result.text`, `result.segments`, and
   `result.language`.
8. Preserve default light FW metadata in `result.backend_metadata`.
9. Preserve FW `words` only when `word_timestamps=true`.
10. Produce SRT only when requested.
11. Update runtime metadata so unsupported/applied override flags are accurate
    for both backend paths.

### `asr-pool-api`

1. Add typed request options for the direct-FW experiment arguments.
2. Serialize those options into `request_json.options` only when set.
3. Keep one `ASRSubmitRequest` type.
4. Keep `ASROutputSelection.text` and `ASROutputSelection.segments`; they
   become real JSON outputs once the pool fills them.
5. Add codec tests for FW options and JSON output selections.

### `asr-translate-tts-dev`

1. Add config for direct-FW experiment options that should be controlled from
   the app.
2. Continue passing `live.asr.backend` through `ASRRequestOptions.asr_backend`.
3. Allow realtime requests to ask for `outputs.text=true` and
   `outputs.segments=true`.
4. Prefer JSON `result.segments` when present; keep SRT parsing only for SRT
   responses.
5. Allow the app to set `vad_filter=false` when its own upstream VAD gate is
   enabled.

## References

- faster-whisper `transcribe.py`:
  https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
- faster-whisper `vad.py`:
  https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py
- WhisperX `asr.py`:
  https://github.com/m-bain/whisperX/blob/main/whisperx/asr.py
