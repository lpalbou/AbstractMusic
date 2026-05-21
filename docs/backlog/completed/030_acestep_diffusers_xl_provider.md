# Completed: ACE-Step Diffusers XL Provider

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-21
- Priority: P1

## Context

ACE-Step is currently the strongest open-source music generation candidate. The existing custom
backend targets `ACE-Step/Ace-Step1.5`, while Hugging Face now has an official Diffusers-format
XL Turbo checkpoint at `ACE-Step/acestep-v15-xl-turbo-diffusers`.

## Current code reality

- `src/abstractmusic/backends/acestep_v15.py` vendors custom ACE-Step Transformers code to avoid
  `trust_remote_code`.
- `src/abstractmusic/backends/diffusers_audio.py` can load generic Diffusers audio pipelines, but
  it maps duration only to `audio_length_in_s` or `audio_end_in_s`.
- The installed local environment on 2026-05-15 had `diffusers 0.38.0`, and
  `diffusers.AceStepPipeline` was available.
- `AceStepPipeline.__call__` accepts `prompt`, `lyrics`, `audio_duration`, `vocal_language`,
  `num_inference_steps`, `guidance_scale`, `shift`, `generator`, and other ACE-Step-specific
  fields.
- The official model card says the Diffusers checkpoint has a standard layout with transformer,
  condition encoder, VAE, text encoder, tokenizer, scheduler, and 48 kHz stereo output.

## Problem

The generic Diffusers backend is too generic for ACE-Step XL. If used unchanged, it will likely
ignore duration and lyrics because the relevant argument names differ from older audio pipelines.

## What we want to do

Add a dedicated `AceStepDiffusersBackend` for
`ACE-Step/acestep-v15-xl-turbo-diffusers`, while keeping the public `MusicManager` API unified.

## Why

The Diffusers-format checkpoint can reduce our custom model-code maintenance burden and provide a
cleaner path to a higher-quality XL model. A dedicated adapter is small, explicit, and safer than
forcing ACE-Step-specific behavior through an overly generic backend.

## Requirements

- Add `AceStepDiffusersBackendConfig` with model id defaulting to
  `ACE-Step/acestep-v15-xl-turbo-diffusers`.
- Lazy-import `torch` and `diffusers`.
- Load `AceStepPipeline` when available; fail with a clear optional dependency/version error when
  Diffusers lacks it.
- Map unified request fields to ACE-Step Diffusers arguments:
  - `prompt` -> `prompt`
  - `lyrics` -> `lyrics`
  - `duration_s` -> `audio_duration`
  - `vocal_language` -> `vocal_language`
  - `num_inference_steps` -> `num_inference_steps`
  - `seed` -> `torch.Generator`
  - `guidance_scale` -> explicit behavior, noting turbo ignores CFG above 1.0 per model card
- Enable VAE tiling when available for long-form audio.
- Encode output to WAV with the stdlib `wave` module or shared audio helpers.
- Return metadata including model id, backend id, duration, sample rate, seed, steps, lyrics flag,
  device, dtype, and fallback information.
- Register the backend in CLI and AbstractCore plugin without making it the only possible model.
- Keep the current custom `AceStepV15Backend` until real smoke tests prove the Diffusers path is
  better as default.

## Suggested implementation

Create a new module, for example `src/abstractmusic/backends/acestep_diffusers.py`, rather than
expanding the generic Diffusers backend. Share WAV encoding helpers only if it reduces duplication
without creating a vague utility layer.

## Scope

- New backend and config.
- CLI/backend selection support.
- AbstractCore plugin support.
- Unit tests with a fake `AceStepPipeline`.
- Documentation updates.

## Non-goals

- Do not implement XL SFT or raw XL Transformers loading in this task.
- Do not require Git-based Diffusers installs if the released package already contains
  `AceStepPipeline`; detect and document the actual minimum supported version.
- Do not add `soundfile` just for examples; WAV output can remain stdlib.

## Dependencies and related tasks

- `docs/backlog/planned/020_music_abstraction_and_capability_registry.md`
- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- Source: `https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers`

## Expected outcomes

- Users can select an ACE-Step Diffusers provider and get WAV bytes through the same
  `MusicManager.t2m(...)` path.
- Lyrics and duration are honored.
- The implementation is much smaller than the custom backend and avoids `trust_remote_code`.

## Validation

- `python -m pytest -q -m "not integration"`
- Unit test with fake `AceStepPipeline` proving `duration_s` maps to `audio_duration` and lyrics
  are passed.
- Optional real smoke test on hardware with enough memory:
  `abstractmusic --backend acestep-diffusers t2m "upbeat synthwave" --duration 5 --out smoke.wav`
- Inspect generated WAV metadata: 48 kHz stereo, nonzero frames, finite PCM, non-silent RMS.

## Progress checklist

- [x] Add backend config and implementation.
- [x] Add CLI and plugin selection.
- [x] Add unit tests.
- [x] Add docs and capability registry entry.
- [x] Run fast tests and at least one real smoke when hardware/model cache permits.

## Progress notes

2026-05-15: Added `AceStepDiffusersBackend`, CLI/backend selection, AbstractCore plugin
registration, and fake-pipeline unit tests. Local `diffusers 0.38.0` exposes `AceStepPipeline`.
The real XL Turbo Diffusers checkpoint is not cached locally beyond metadata yet, so a real XL
smoke remains pending.

## Guidance for the implementing agent

Do not hide ACE-Step-specific parameters inside a generic backend. The point of this task is a
small explicit adapter under a stable public abstraction.

## Completion report

2026-05-21:

- The dedicated `acestep-diffusers` backend is wired end-to-end (CLI + AbstractCore plugin) and
  honors duration/lyrics by mapping to `AceStepPipeline` parameters (`audio_duration`, `lyrics`,
  etc).
- Real smoke generation has been exercised locally:
  - `test-artifacts/real-generation/acestep_smoke.wav`
  - `untracked/duration-check/acestep-diffusers-25.wav` (25s duration check harness)

Touched:

- `src/abstractmusic/backends/acestep_diffusers.py`
- `src/abstractmusic/cli.py`
- `src/abstractmusic/integrations/abstractcore_plugin.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- `tests/test_acestep_diffusers_backend.py`

Validation:

- `python -m pytest -q`
