# Planned: MusicGen Small Optional Provider

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
- Priority: P1

## Context

ACE-Step is technically wired through the unified AbstractMusic interface, but recent listening
feedback and signal checks show repetitive output collapse on Apple MLX/MPS. We need a small,
known text-to-music baseline to separate "our ACE-Step integration is wrong" from "this model path
is not good enough for the target use case."

`facebook/musicgen-small` is a 300M text-to-music checkpoint with a documented Transformers
inference path. It is not commercially usable because the model weights are CC BY-NC 4.0, but it
is valuable as a small quality and abstraction benchmark.

## Current code reality

- The model registry tracks `facebook/musicgen-small` as an implemented-unvalidated validation
  candidate.
- `pyproject.toml` has a concrete `musicgen` extra with the lightweight Transformers runtime.
- `src/abstractmusic/backends/musicgen.py` implements the provider through
  `AutoProcessor` and `MusicgenForConditionalGeneration`.
- The CLI/REPL route `--engine musicgen` and aliases such as `musicgen-small`.
- AbstractMusic already has a unified request/result contract, CLI/REPL, model registry, and WAV
  artifact inspection.
- The generic Diffusers backend is not enough for MusicGen because MusicGen is exposed through
  Transformers text-to-audio/model classes rather than a Diffusers pipeline.

## Problem

We currently have no small, stable baseline that can quickly prove the AbstractMusic user flow
with recognizable music. ACE-Step remains the preferred open/commercial candidate, but it is not
yet passing subjective quality review.

## What We Want To Do

Add an optional `musicgen` backend that can generate short WAV files from text prompts through
`facebook/musicgen-small`, then validate whether the output is recognizably musical and less
repetitive than the current ACE-Step Apple path.

## Why

This gives the project a pragmatic comparison point with minimal conceptual complexity. If
MusicGen small produces coherent short clips through the same CLI/REPL, then the abstraction is
healthy and ACE-Step quality work can be isolated to ACE-Step.

## Requirements

- Keep the base install dependency-free; do not add MusicGen dependencies to base.
- Prefer the plain Transformers path over the heavier Audiocraft runtime.
- Support at least `prompt`, `duration_s`, `seed`, and WAV output.
- Use Apple MPS when PyTorch supports the path, CUDA on GPU machines, and CPU only as fallback.
- Return metadata including model id, duration, seed, sample rate, backend id, and quality metrics.
- Mark the backend non-commercial in capabilities and do not make it the production default.
- Do not require SciPy if existing NumPy/WAV helpers can write valid PCM directly.

## Suggested Implementation

Create `src/abstractmusic/backends/musicgen.py` with a `MusicGenBackendConfig` and
`MusicGenBackend`. Use `AutoProcessor` and `MusicgenForConditionalGeneration` from Transformers.
Calculate `max_new_tokens` from requested duration using MusicGen's 50 Hz token rate when the
model API does not expose a direct duration parameter. Reuse AbstractMusic's WAV conversion helpers
or add a small internal tensor-to-WAV helper instead of pulling an audio writer dependency.

Add CLI/backend routing aliases (`musicgen`, `musicgen-small`) and registry tests. Then run a real
10-second smoke and record metrics plus a listening note before changing any recommendation status.

## Scope

- `src/abstractmusic/backends/musicgen.py`
- `src/abstractmusic/cli.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- `pyproject.toml` optional `musicgen` extra
- Focused unit tests and one opt-in real generation test
- `docs/models.md` and backlog notes

## Non-Goals

- Do not make MusicGen the default provider because its weights are non-commercial.
- Do not add Audiocraft unless Transformers cannot satisfy the provider contract.
- Do not implement melody-conditioned MusicGen in this item.

## Dependencies And Related Tasks

- `docs/backlog/planned/045_audio_artifact_screening_and_quality_metadata.md`
- `docs/backlog/planned/065_acestep_repetition_quality_gate.md`
- `src/abstractmusic/backends/acestep_official.py`
- `src/abstractmusic/audio_analysis.py`

## Expected Outcomes

AbstractMusic has a small non-commercial provider that can be used as a real generation baseline
for the CLI/REPL and quality metrics. ACE-Step can then be judged against a known working small
music model rather than in isolation.

## Validation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_model_capabilities.py tests/test_cli_smoke.py`
- Opt-in real smoke: 10-second prompt-generated WAV through `facebook/musicgen-small`.
- Inspect generated WAV for non-silence, harmonic structure, clipping, and repetition.
- Human listening spot-check before calling it "working music."

## Progress Checklist

- [x] Re-check current MusicGen Transformers API and license before editing.
- [x] Implement optional backend with lazy imports and no base dependency changes.
- [x] Add CLI/REPL routing and model capability tests.
- [x] Run focused tests.
- [ ] Run and document one real 10-second generation smoke.
- [ ] Update recommendation status based on validation.

## Progress Notes

2026-05-16: Implemented the optional `musicgen` backend, CLI/REPL routing, concrete
`musicgen` dependency extra, registry metadata, and unit tests. Real 10-second generation remains
pending before this can be considered a working music provider.

## Guidance For The Implementing Agent

Treat this as a benchmark provider, not as a default product answer. Keep the implementation small,
validate real audio, and preserve the provider abstraction so ACE-Step, MusicGen, and future
models remain interchangeable from the caller's perspective.
