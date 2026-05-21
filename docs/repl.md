# REPL Guide

The AbstractMusic REPL is an interactive shell for iterating on prompts, engines, models, and
generation parameters without restarting the process.

Launch it with either command (they are the same):

```bash
abstractmusic cli
abstractmusic repl
```

You can also set the starting engine and output directory:

```bash
abstractmusic cli --engine acemusic --duration 25 --out-dir smoke-artifacts/repl
abstractmusic cli --engine acestep --duration 25 --out-dir smoke-artifacts/repl
```

## Core concepts

- **Engine**: the backend implementation (`acemusic`, `acestep-diffusers`, `stable-audio-3`, …).
- **Model**: the provider model id from the packaged registry (for local engines, this is usually a
  Hugging Face repo id).

The REPL keeps an **engine + model** pair. If you pick a known model with `/model`, the REPL will
auto-switch to the correct engine when it can.

## Quick start

```text
/status
/prompt high-energy electronic synthwave track with heavy bass
/run
```

## Styles, instruments, and vocal traits

Use style tags to steer instrumentation and vocal character without rewriting your entire prompt.

```text
/style saxophone, brushed drums, female vocalist, smoky jazz club
/negative-style no autotune, no chipmunk voice
/prompt a clean jazz arrangement with a clear chorus lift
/run
```

Notes:

- For `elevenlabs`, style tags are passed as a provider-neutral `composition_plan` (so it will run in plan mode).
- For other engines, style tags are appended into the prompt text as lightweight “tags”.
- This is **not** voice cloning or a guaranteed vocalist identity selector. For explicit voice selection / cloning, use AbstractVoice.

## Discover engines and models

- `/engines` lists engines and their packaged default models.
- `/models` lists all known models, grouped by engine.
- `/model` shows the current model selection (or which default is in use).
- `/engine` shows the current engine selection (and the effective model).

Examples:

```text
/engines
/models
/engine stable-audio-3
/model stabilityai/stable-audio-3-medium
/status
```

## Switching engines and models (important)

Use `/model <id>` when you know which model you want. It will:

- set the model id
- auto-switch the engine if the model is known
- auto-clamp `/duration` if the model has a max duration (ex: Stable Audio Open Small)

Use `/engine <name>` when you want to force a backend implementation. If the current explicit model
belongs to a different engine, the REPL clears it back to that engine’s default to avoid accidental
mismatches.

## Downloads / offline mode

Local engines load Hugging Face weights. By default, downloads are **enabled** unless any of these
are set:

- `ABSTRACTMUSIC_LOCAL_FILES_ONLY=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `DIFFUSERS_OFFLINE=1`

Within a REPL session, use:

```text
/download off
```

to require cache-only loading (useful for offline runs), or:

```text
/download on
```

to allow pulling missing files.

## Duration limits

- `stable-audio` (Stable Audio Open Small) supports **up to 11 seconds**.
- `stable-audio-3` models support longer durations (Small Music up to 120s; Medium up to 380s).

If you request too long a duration for a model, `/model` may auto-adjust the current `/duration`
down to the model maximum.

## Common commands

```text
/help
/status
/prompt <text|clear>
/run
/duration <seconds>
/style <tags|clear>
/negative-style <tags|clear>
/steps <n|auto>
/seed <n|auto>
/format <wav|mp3|flac>
/engine <name>
/model <id|default|clear>
/models
/engines
/params
/exit
```
