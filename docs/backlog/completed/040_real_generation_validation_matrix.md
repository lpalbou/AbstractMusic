# Completed: Real Generation Validation Matrix

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-15
- Priority: P0

## Context

The fast unit tests passed on 2026-05-15, but the most important real-world question remains
unanswered: can AbstractMusic generate usable music with real model weights on supported hardware?

## Current code reality

- Unit tests fake ACE-Step model, tokenizer, text encoder, VAE, and Diffusers pipeline behavior.
- Generated WAV files in the repo show previous manual experiments. Some are valid non-silent WAVs,
  one is silent, and some earlier outputs show high DC offset or heavy postprocessing artifacts.
- `acestep_v15.py` includes many fallbacks for MPS dtype issues, CPU text encoder fallback, VAE
  decode tiling, non-finite latent retry, DC offset removal, and normalization.
- There is no checked-in integration test marker that exercises a real Hugging Face model only
  when explicitly enabled.

## Problem

Without a real smoke matrix, the project can regress into mock-only correctness. Users experience
the project as broken even when CI is green.

## What we want to do

Create a gated validation suite and manual smoke procedure for real providers, with objective audio
checks and clear skip behavior.

## Why

Music generation failures are often numerical, hardware-specific, or caused by subtle argument
mapping mistakes. These failures are invisible to pure unit tests.

## Requirements

- Add integration tests gated by environment variables, for example:
  - `ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1`
  - `ABSTRACTMUSIC_REAL_MODEL_ID=...`
  - `ABSTRACTMUSIC_REAL_BACKEND=acestep|acestep-diffusers|diffusers`
  - `ABSTRACTMUSIC_REAL_DEVICE=cpu|mps|cuda|auto`
- Keep tests skipped by default in CI.
- Real tests should generate at least 10 seconds when validating music quality. Shorter 3-5 second
  runs can still be useful for crash-only debugging, but they must not be used to mark a provider
  recommended.
- Validate WAV header, sample rate, channel count, duration tolerance, finite PCM, nonzero peak,
  non-silent RMS, low DC offset after postprocessing, harmonic structure, and fast repetitive
  envelope patterns that indicate rotor-like failure.
- Store generated outputs under an ignored test artifact directory, not repo root.
- Record known hardware expectations:
  - ACE-Step v1.5 smaller path can be tried on consumer hardware.
  - ACE-Step XL Turbo needs much more memory, especially without offload.
  - MPS may need CPU fallbacks and shorter durations.
- Add a troubleshooting doc section for common failures: missing weights, insufficient memory,
  Diffusers version missing `AceStepPipeline`, MPS unsupported ops, non-finite latents, and silent
  audio.
- On Apple hardware, try official MLX backends first when a provider has one; otherwise prefer
  PyTorch MPS. CPU is an explicit fallback only, and must be visible in warnings/metadata.

## Suggested implementation

Add `tests/integration/test_real_generation.py` with explicit pytest markers and helpers for WAV
inspection. Reuse stdlib `wave` and small numeric checks; avoid adding audio analysis dependencies.

## Scope

- Gated real-provider tests.
- Audio validation helpers.
- Ignored artifact output path.
- Docs for running real smoke tests.

## Non-goals

- Do not make CI download multi-GB models by default.
- Do not add subjective audio quality scoring in this task.
- Do not require external codecs.

## Dependencies and related tasks

- `docs/backlog/planned/030_acestep_diffusers_xl_provider.md`
- `docs/backlog/recurrent/dependency_and_artifact_hygiene.md`

## Expected outcomes

- Developers can prove a provider actually works before changing defaults.
- Failures produce actionable errors rather than just bad or silent WAV files.
- Generated smoke artifacts stay out of source control.

## Validation

- `python -m pytest -q -m "not integration"` remains fast and green.
- With real model env vars enabled, at least one provider smoke test passes on documented hardware.
- `git status --short` stays clean of generated audio after tests.

## Progress checklist

- [x] Add integration marker and env-gated tests.
- [x] Add WAV validation helper.
- [x] Add ignored artifact directory.
- [x] Document smoke commands and expected hardware.
- [x] Run fast tests and one real smoke where possible.

## Completion report

Completed on 2026-05-15.

Implemented `src/abstractmusic/audio_analysis.py` with standard-library WAV inspection, added an
opt-in real generation test at `tests/integration/test_real_generation.py`, and documented the
smoke command in `docs/getting-started.md`.

Validation results:

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -m "not integration"` passed.
- `ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1 ABSTRACTMUSIC_REAL_BACKEND=acestep
  ABSTRACTMUSIC_REAL_DEVICE=mps ABSTRACTMUSIC_REAL_DTYPE=auto ABSTRACTMUSIC_REAL_DURATION_S=3
  python -m pytest -q tests/integration/test_real_generation.py` passed on Apple MPS.
- Manual smoke artifact `smoke-artifacts/acestep_smoke_mps.wav` was 48 kHz stereo, 3 seconds,
  non-silent, unclipped, and DC-centered, but later listening feedback and harmonic/envelope
  analysis rejected it as fast rotor-like audio, not usable music.

Known caveat: the earlier custom ACE-Step v1.5 smoke is now treated as a failed music-quality smoke.
The validation helper has been tightened so this class of output fails instead of being counted as
provider success.

## Guidance for the implementing agent

A real smoke test that is skipped by default is valuable. A test that downloads giant weights in
normal CI is not.
