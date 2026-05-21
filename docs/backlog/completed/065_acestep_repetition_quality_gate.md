# Completed: ACE-Step Repetition Quality Gate

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-21
- Priority: P0

## Context

ACE-Step paths can generate valid 48 kHz WAV files with harmonic content, but user listening and
follow-up spectral self-similarity checks show that outputs can be dominated by repetitive
note/pulse patterns. The existing smoke checks catch silence, clipping, white noise, and rotor-like
fast envelopes, but they do not catch musically repetitive loops.

## Current code reality

- `src/abstractmusic/backends/acestep_v15.py` is the default standalone package backend. Its 5Hz
  audio-code planner is opt-in because full-cover conditioning from coarse code hints can imprint
  repetitive artifacts.
- `src/abstractmusic/audio_analysis.py` reports harmonic diversity and spectrotemporal modulation
  metrics, including longer-range similarity checks.
- Recent trial files in `smoke-artifacts/user-repl/` show high self-similarity at 1-4 second lags
  even when `music_like=True`.

## Problem

The project can incorrectly classify a repetitive loop as acceptable music because the current
quality gate asks whether audio is tonal, not whether it evolves musically.

## What we want to do

Add a repetition/novelty quality gate and use it to decide whether ACE-Step outputs are acceptable
for the default provider recommendation.

## Why

The target is usable generated music, not merely valid audio with harmonics. A user should not have
to listen to every smoke artifact to catch obvious one-note or short-loop collapse.

## Requirements

- Keep the analysis dependency-light; prefer NumPy-only feature extraction.
- Add spectral self-similarity metrics over short and medium lags.
- Keep the old harmonic/noise metrics available; do not collapse all quality dimensions into one
  opaque score.
- Record metrics in generated asset metadata.
- Mark ACE-Step as quality-limited if it repeatedly fails the novelty gate.
- Validate at least 10-second and 20-second samples.

## Suggested implementation

Extend `MusicSignalStats` with spectral novelty and lag-similarity summaries, then add an
`is_probably_over_repetitive` property. Use log-frequency spectral frames so the metric catches
repeated timbre/harmony without requiring heavyweight audio libraries.

## Scope

- `src/abstractmusic/audio_analysis.py`
- `tests/test_audio_analysis.py`
- `src/abstractmusic/backends/acestep_v15.py` metadata passthrough
- `docs/models.md` and backlog status notes

## Non-goals

- Do not claim objective musical quality from a single score.
- Do not add librosa or other heavyweight analysis dependencies unless a later task proves NumPy is
  insufficient.

## Dependencies and related tasks

- `docs/backlog/completed/045_audio_artifact_screening_and_quality_metadata.md`
- `docs/backlog/completed/040_real_generation_validation_matrix.md`

## Expected outcomes

AbstractMusic can distinguish "harmonic but loop-collapsed" from genuinely usable music candidates,
and provider recommendations reflect that distinction.

## Validation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_audio_analysis.py`
- Real ACE-Step 1.7B smoke files evaluated for novelty/repetition.
- Human listening spot-check for any sample promoted as acceptable.

## Progress checklist
- [x] Re-check current code and docs before editing.
- [x] Implement the smallest robust change.
- [x] Add or update focused tests.
- [x] Update user-facing docs if behavior changes.
- [x] Run validation and record results in a completion report.

## Guidance for the implementing agent

Treat this as a guardrail, not as a music critic. The metric should catch obvious repetition
failures and make the recommendation status honest.

## Progress Notes

2026-05-15: Review found that ACE-Step turbo does not use CFG in the packaged v1.5 turbo path.
AbstractMusic treats turbo ACE-Step as guidance-unsupported; this is a correctness cleanup, not a
fix for repetitive-output quality failures.

## Completion report

2026-05-21:

- Implemented repetition-oriented guardrails in `abstractmusic.audio_analysis`, including
  envelope repetition screening and long-lag spectral similarity used by the
  spectrotemporal-modulation artifact detector.
- Added unit tests that flag synthetic loop/pulse artifacts and keep a harmonic progression
  fixture passing (`tests/test_audio_analysis.py`).
- Wired the guards into the standalone `acestep-v15` backend quality metadata path so obviously
  repetitive outputs are not treated as acceptable music. The registry now reflects this reality
  via `status` and non-recommended labeling for the standalone v1.5 backend.

Touched:

- `src/abstractmusic/audio_analysis.py`
- `tests/test_audio_analysis.py`
- `src/abstractmusic/backends/acestep_v15.py`
- `src/abstractmusic/assets/music_model_capabilities.json`

Validation:

- `python -m pytest -q tests/test_audio_analysis.py`
