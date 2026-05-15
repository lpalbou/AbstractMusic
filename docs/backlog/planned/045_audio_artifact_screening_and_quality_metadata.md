# Planned: Audio Artifact Screening And Quality Metadata

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
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

- [ ] Add envelope autocorrelation and repeated-chunk metrics.
- [ ] Add first-class quality metadata fields to the registry.
- [ ] Add recommendation guard tests.
- [ ] Update backend asset metadata with quality-smoke metrics.
