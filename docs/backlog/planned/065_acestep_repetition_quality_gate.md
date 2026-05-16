# Planned: ACE-Step Repetition Quality Gate

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
- Priority: P0

## Context

The official ACE-Step path can generate valid 48 kHz WAV files with harmonic content, but user
listening and follow-up spectral self-similarity checks show that short instrumental outputs can be
dominated by repetitive note/pulse patterns. The existing smoke checks catch silence, clipping,
white noise, and rotor-like fast envelopes, but they do not catch musically repetitive loops.

## Current code reality

- `src/abstractmusic/backends/acestep_official.py` wraps upstream ACE-Step and now exposes the
  important turbo controls: 1.7B LM default, `shift=3.0`, LM sampling controls, and
  `audio_cover_strength`.
- `src/abstractmusic/audio_analysis.py` reports harmonic and fast-envelope metrics, but it does
  not report longer-range spectral self-similarity or musical novelty.
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
- `src/abstractmusic/backends/acestep_official.py` metadata passthrough
- `docs/models.md` and backlog status notes

## Non-goals

- Do not claim objective musical quality from a single score.
- Do not add librosa or other heavyweight analysis dependencies unless a later task proves NumPy is
  insufficient.

## Dependencies and related tasks

- `docs/backlog/planned/045_audio_artifact_screening_and_quality_metadata.md`
- `docs/backlog/completed/025_official_acestep_v15_mlx_provider.md`

## Expected outcomes

AbstractMusic can distinguish "harmonic but loop-collapsed" from genuinely usable music candidates,
and provider recommendations reflect that distinction.

## Validation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_audio_analysis.py`
- Real ACE-Step 1.7B smoke files evaluated for novelty/repetition.
- Human listening spot-check for any sample promoted as acceptable.

## Progress checklist
- [ ] Re-check current code and docs before editing.
- [ ] Implement the smallest robust change.
- [ ] Add or update focused tests.
- [ ] Update user-facing docs if behavior changes.
- [ ] Run validation and record results in a completion report.

## Guidance for the implementing agent

Treat this as a guardrail, not as a music critic. The metric should catch obvious repetition
failures and make the recommendation status honest.

## Progress Notes

2026-05-15: Online/upstream review found that ACE-Step turbo explicitly does not use CFG and
upstream clamps turbo `guidance_scale` to `1.0` to avoid noisy/NaN float16 behavior. AbstractMusic
now treats the default official turbo path as guidance-unsupported and defaults turbo guidance to
`1.0`; this is a correctness cleanup, not a fix for the repetitive-output quality failure.
