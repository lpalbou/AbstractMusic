# Proposed: The CLI never shows the device and dtype it used; Stable Audio runs float32 without a reason and overshoots the duration

## Metadata
- Created: 2026-11-10
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context
Found during the 0.14.0 install-profile proof (abstractmusic 0.1.17) on three hosts:

- Apple silicon, ACE-Step turbo through the CLI (`abstractmusic --backend acestep t2m "a short calm piano loop"
  --out music.wav --duration 8 --device auto --verbose --print-plan`): the CLI prints the plan and the output path
  only. The Python API reports `device: mps`, `dtype: bfloat16`, load 15.8 s, about 3 s per 8 s clip; none of it
  reaches the CLI user.
- Linux NVIDIA (Quadro RTX 5000), MusicGen small through the CLI with `--device cuda`: the same, output path only;
  whether the card was used is visible only through `nvidia-smi` (it was: 1.7 GB, 51 % peak).
- Linux processor-only, MusicGen small: the same; device appears only in the result metadata, dtype nowhere.
- Apple silicon, Stable Audio Open Small called directly (`MusicManager(backend=StableAudioBackend(...))`): runs
  on MPS in float32, the checkpoint's stored dtype, with no dtype choice, no report and no reason (ACE-Step on
  the same machine picks and reports bfloat16); asked for 8 s it returned an 11.89 s file with 32.7 % near-silent
  frames while its metadata says `duration_s: 8.0`.

## Current code reality
`cli.py` `t2m` writes the output path to stdout; the backend result metadata (device, dtype, fallback_events,
timings) is not printed, even with `--verbose`. The Stable Audio backend loads the checkpoint at its stored dtype
and does not trim or pad the waveform to the requested duration.

## Problem
A CLI user cannot tell whether a generation ran on the GPU or on the processor, nor in which precision; the Stable
Audio result does not match its own metadata.

## What we want to do
- `t2m` (and `i2m`/`repl` where it applies) prints one line after the output path: backend, model, device, dtype,
  load and generation time, audio seconds and sample rate; `--json` returns the full metadata.
- Stable Audio: pick the dtype with the same policy as the other local backends (float16 on CUDA/MPS, float32 on
  the processor, bfloat16 where the package prefers it) and report it with a reason; trim or pad the waveform to
  the requested duration or report the real one.

## Why
"Optimal" on each platform must be visible to the user from the CLI; a file that is 50 % longer than requested
with a silent tail is not the requested clip.

## Requirements
- One CLI test asserting the device/dtype line against a stub backend.
- One Stable Audio test on the duration (metadata equals the file) and the dtype report.

## Suggested implementation
Reuse the metadata the backends already return (`device`, `dtype`, `fallback_events`); format it in `cli.py`.
For Stable Audio, route the dtype through `diffusers_audio._resolve_dtype` and cut the waveform at
`duration_s * sample_rate`.

## Scope
`cli.py`, the Stable Audio backend, two tests, README line for the new output.

## Non-goals
Changing which backend is the default.

## Dependencies and related tasks
- 0091 (the documented Stable Audio CLI command fails)

## Expected outcomes
Every CLI generation says where and how it ran; Stable Audio output matches its metadata.
