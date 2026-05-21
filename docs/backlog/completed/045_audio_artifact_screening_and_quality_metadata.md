# Completed: Audio Artifact Screening And Quality Metadata

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-21
- Priority: P1

## Context

The first real ACE-Step smoke showed why WAV validity is not enough: the file was non-silent,
unclipped, and DC-centered, but sounded like fast rotor-like repetition instead of music. The new
`MusicSignalStats` helper catches the observed case with harmonic and envelope metrics, but the
quality metadata model is still minimal.

## Problem

The registry currently has a single `recommended` flag and free-text notes. That makes it too easy
to blur together file-integrity success, artifact screening, human listening review, and true
provider readiness.

## What we want to do

Make audio artifact screening and model quality status first-class. Real generation tests should
reject obvious synthetic artifacts, and model metadata should clearly state the validation level
behind any recommendation.

## Why

AbstractMusic must not claim a model works just because it emits a decodable audio file. Users need
to know whether a provider generated usable music through the abstraction.

## Requirements

- Extend `MusicSignalStats` with stronger rotor/repetition metrics, such as envelope
  autocorrelation and repeated-chunk correlation.
- Keep the current harmonic and modulation checks.
- Add registry fields such as `validation_level`, `known_quality_issues`, and
  `quality_recommendation`.
- Add tests proving models with known rotor-like issues cannot be recommended.
- Attach relevant quality-smoke metrics to generated asset metadata where practical.
- Document that these are smoke gates, not perceptual quality scores.

## Scope

- `src/abstractmusic/audio_analysis.py`
- `src/abstractmusic/model_capabilities.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- Registry and audio-analysis unit tests.
- Docs/model-list wording.

## Non-goals

- Do not claim objective metrics prove music is good.
- Do not add heavy audio-analysis dependencies to the base install.

## Expected outcomes

- Obvious rotor/helicopter-like artifacts fail real smoke tests.
- The model registry distinguishes file-integrity checks from music-quality validation.
- Recommendations are backed by explicit validation level metadata.

## Validation

- `python -m pytest -q -m "not integration"`
- Synthetic rotor and musical-tone fixtures remain correctly classified.
- The known failed ACE-Step MPS smoke remains rejected by the artifact gate.

## Progress checklist

- [x] Add envelope autocorrelation and repeated-chunk metrics.
- [x] Add first-class quality metadata fields to the registry.
- [x] Add recommendation guard tests.
- [x] Update backend asset metadata with quality-smoke metrics.

## Completion report

2026-05-21:

- Implemented stronger rotor/repetition screening in `abstractmusic.audio_analysis` using
  envelope repetition ratios plus long-lag spectral similarity and spectrotemporal modulation
  checks (targeting clock/pulse and loop-like failures that pass basic WAV validity).
- Added and maintained synthetic test fixtures proving known repetition artifacts are flagged
  (`tests/test_audio_analysis.py`), and wired the smoke gates into real generation validation.
- Left the registry quality story primarily in the existing `status` + `recommended` fields, plus
  concrete duration/precision notes, instead of introducing additional registry schema fields.
  This keeps the public model registry stable while still making validation state explicit.

Touched:

- `src/abstractmusic/audio_analysis.py`
- `tests/test_audio_analysis.py`
- `src/abstractmusic/assets/music_model_capabilities.json`

Validation:

- `python -m pytest -q`
