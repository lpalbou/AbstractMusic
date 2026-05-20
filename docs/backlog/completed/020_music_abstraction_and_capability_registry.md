# Planned: Music Abstraction And Capability Registry

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-15
- Priority: P0

## Context

The user explicitly requires an abstraction over different models so AbstractMusic provides a
unified way to generate music. The current protocol is too small to express model capability,
provider catalogs, lifecycle hooks, or unsupported parameter behavior.

## Current code reality

- `MusicBackend` only requires `backend_id` and `generate_audio(request)`.
- `AudioGenerationRequest` has prompt, negative prompt, duration, steps, guidance scale, seed, and
  `extra`.
- Lyrics are currently passed through `extra` from `MusicManager.generate_audio`.
- `GeneratedAsset` lacks an explicit `media_type` field.
- The AbstractCore plugin has separate ACE-Step and Diffusers capability wrappers but no
  capability registry comparable to AbstractVision.

## Problem

Without a richer contract, every provider must guess how to handle fields like lyrics, duration,
guidance, language, sample rate, or future video/reference audio. That leads to silent feature loss
or model-specific public APIs.

## What we want to do

Create a music provider contract and capability registry that can represent ACE-Step, generic
Diffusers audio, TinyMozart-style MIDI, and future multimodal providers without making the public
API unstable.

## Why

A clear abstraction lets AbstractCore and application users choose models intentionally, detect
unsupported features early, and keep model-specific logic inside backends.

## Requirements

- Add `ProviderModelInfo` and `MusicBackendCapabilities` types.
- Add first-class request fields where they are common enough: `lyrics`, `vocal_language`,
  `sample_rate`, `format`, and possibly `reference_audio`.
- Keep more specialized fields in explicit typed extras only when necessary.
- Add optional backend methods: `get_capabilities()`, `list_provider_models()`, `preload()`,
  `unload()`, and progress-capable generation if useful.
- Add `src/abstractmusic/assets/music_model_capabilities.json` and a loader/validator.
- Represent at least these model families in metadata:
  - `ACE-Step/Ace-Step1.5`
  - `ACE-Step/acestep-v15-xl-turbo-diffusers`
  - `ACE-Step/acestep-v15-xl-turbo`
  - `ACE-Step/acestep-v15-xl-sft`
  - `LH-Tech-AI/TinyMozart_v2_85M`
  - `Dalision/Omni2Sound`
- Metadata should include license, commercial suitability, provider kind, input modalities,
  output formats, lyrics support, guidance support, max duration if known, dependency extra, and
  notes.
- Do not silently select a different provider or model from capability metadata.

## Suggested implementation

Mirror AbstractVision's shape: dataclasses in `types.py`, an abstract or protocol backend in
`backends/base_backend.py`, and a registry loader similar to
`abstractvision.model_capabilities`. Keep imports light and JSON parsing stdlib-only.

## Scope

- Public types and backend protocol.
- Packaged model capability registry.
- Manager-side capability validation for common tasks.
- Tests for schema loading, unsupported fields, and plugin metadata.

## Non-goals

- Do not add real model inference here.
- Do not add video input support beyond reserving clean extension points unless a provider is
  implemented in the same task.

## Dependencies and related tasks

- `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- `docs/backlog/planned/030_acestep_diffusers_xl_provider.md`
- `docs/backlog/completed/050_dependency_profiles_and_optional_providers.md`

## Expected outcomes

- Callers can inspect what a configured backend supports.
- The manager can reject unsupported parameters instead of letting backends ignore them.
- New providers have a predictable integration checklist.

## Validation

- `python -m pytest -q -m "not integration"`
- Unit tests for registry loading and required model entries.
- Unit tests proving unsupported lyrics/guidance/negative prompt behavior is explicit.
- Import test proving `import abstractmusic` does not import Torch/Diffusers.

## Progress checklist

- [ ] Extend request/result/capability types.
- [ ] Extend backend protocol.
- [ ] Add model capability registry asset and loader.
- [ ] Wire manager validation without breaking current public calls.
- [ ] Update docs and tests.

## Guidance for the implementing agent

Keep the abstraction small and real. Add fields because multiple providers or callers need them,
not because one backend happens to expose a long parameter list.

## Completion report

### Summary

Implemented the first version of the music model abstraction and capability registry. The public
request/result types now carry common music fields explicitly, backends can report capabilities,
and AbstractMusic has a packaged model registry covering ACE-Step, HeartMuLa, TinyMozart, and
Omni2Sound.

### Files and symbols touched

- Added `src/abstractmusic/model_capabilities.py`
- Added `src/abstractmusic/assets/music_model_capabilities.json`
- Added `src/abstractmusic/assets/__init__.py`
- Updated `src/abstractmusic/types.py` with `ProviderModelInfo`, `MusicBackendCapabilities`,
  `lyrics`, `vocal_language`, `format`, `sample_rate`, and `GeneratedAsset.media_type`
- Updated `src/abstractmusic/backends/base_backend.py` with optional capability/catalog/lifecycle
  methods
- Updated `src/abstractmusic/music_manager.py` with model/backend capability validation
- Updated ACE-Step and generic Diffusers backends to expose known capabilities
- Added `tests/test_model_capabilities.py`

### Validation

- `python -m pytest -q -m "not integration"` passed with 22 tests.
- `python -m build && python -m twine check dist/*` passed.
- Added a subprocess test proving `import abstractmusic` does not import `torch`, `diffusers`, or
  `transformers`.

### Residual risks

The registry is metadata only. It does not prove real model generation works. HeartMuLa is listed
as a planned research provider and still needs a dedicated optional backend task.
