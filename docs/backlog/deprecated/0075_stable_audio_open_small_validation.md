# Deprecated: Stable Audio Open Small Validation

## Metadata
- Created: 2026-05-16
- Status: Deprecated
- Completed: N/A
- Deprecated: 2026-05-21
- Priority: P2

## Context

`stabilityai/stable-audio-open-small` is a small gated Stable Audio model intended for short
text-to-audio clips. It is interesting as a small local comparison point, especially for short
loops and sound effects, but it is not a strong default music candidate and requires Hugging Face
access approval.

## Current Code Reality

- The model registry tracks `stabilityai/stable-audio-open-small` as
  `configured-unvalidated-gated`.
- `pyproject.toml` has a concrete `stable-audio` extra.
- `src/abstractmusic/backends/stable_audio.py` uses AbstractMusic-vendored `stable-audio-tools==0.0.19`
  model code and a minimal rectified-flow inference loop (no upstream package install required).
- The CLI/REPL route `--engine stable-audio` and aliases such as `stable-audio-open-small`.
- The AbstractCore integration plugin registers `abstractmusic:stable-audio` (see
  `docs/backlog/completed/0085_truthful_stable_audio_capability_registration_and_music_routing.md`),
  but the model remains gated and unvalidated.

## Problem

The backend is configured and capability routing is now truthful, but the provider remains gated
and should not be treated as validated until a machine with accepted Hugging Face model access can
generate and inspect a real WAV.

## What We Want To Do

Validate one 10-11 second generation through AbstractMusic, capture objective audio metrics, and
record whether the model is useful for music, sound effects, or neither.

## Why

Stable Audio Open Small is small enough to be worth testing, but gated licensing and music-quality
limits mean it should not be promoted without evidence.

## Requirements

- Keep the provider optional and non-default.
- Do not add Stable Audio dependencies to the base install.
- Require explicit Hugging Face access approval/token when weights are not available.
- Validate WAV structure, clipping, harmonic/noise metrics, and repetition.
- Human listening spot-check before marking it as usable.

## Suggested Implementation

Run:

```bash
abstractmusic --engine stable-audio t2m "short ambient synth loop" --duration 10 --steps 8 --out smoke-artifacts/stable-audio-open-small.wav
```

If access fails, document the exact gated-access failure and leave status as configured-unvalidated.

## Scope

- Runtime validation only.
- `docs/models.md`
- `src/abstractmusic/assets/music_model_capabilities.json`
- Backlog completion notes if validation succeeds or fails conclusively.

## Non-Goals

- Do not make this a default provider.
- Do not bypass Hugging Face license gating.
- Do not claim full music generation quality from a sound-effect-oriented short clip.

## Dependencies And Related Tasks

- `docs/backlog/completed/0085_truthful_stable_audio_capability_registration_and_music_routing.md`
- `docs/backlog/completed/045_audio_artifact_screening_and_quality_metadata.md`
- `docs/backlog/completed/065_acestep_repetition_quality_gate.md`

## Expected Outcomes

Stable Audio Open Small is either validated as a useful optional short-clip provider or explicitly
documented as gated/unusable in the current environment.

## Validation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_stable_audio_backend.py tests/test_model_capabilities.py tests/test_cli_smoke.py`
- Real gated-model smoke when HF access is available.

## Progress Checklist

- [x] Implement optional backend with lazy imports and no base dependency changes.
- [x] Add CLI/REPL routing and model capability tests.
- [x] Add concrete optional dependency extra.
- [ ] Install provider extra in the local venv.
- [ ] Run real generation smoke or record gated-access blocker.
- [ ] Update recommendation status based on validation.

## Guidance For The Implementing Agent

Treat this as a short-clip provider until proven otherwise. The model card and license/access
boundary matter as much as the code path.

## Deprecation report

2026-05-21:

This validation item is no longer tracked as a separate planned workstream.

- The Stable Audio Open Small backend remains available as an optional, gated, short-clip provider
  (`abstractmusic:stable-audio`), but it is not a priority compared to the Stable Audio 3 track.
- The capability-boundary gap that previously blocked higher-layer selection is closed by
  `../completed/0085_truthful_stable_audio_capability_registration_and_music_routing.md`.
- Any future Open Small validation should be folded into broader provider-validation work rather
  than revived as a standalone planned item.
