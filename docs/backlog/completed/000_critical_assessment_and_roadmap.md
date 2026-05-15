# Planned: Critical Assessment And Roadmap

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-15
- Priority: P0

## Context

The project has had several attempts at real local music generation. The current tree contains a
working-looking abstraction and many ACE-Step compatibility patches, but the project is still not
trustworthy as a usable package because real provider validation and repository hygiene are missing.

## Current code reality

- `src/abstractmusic/music_manager.py` has a thin `MusicManager` facade.
- `src/abstractmusic/types.py` defines only `AudioGenerationRequest` and `GeneratedAsset`.
- `src/abstractmusic/backends/base_backend.py` defines a one-method `MusicBackend` protocol.
- `src/abstractmusic/backends/acestep_v15.py` contains a large in-process ACE-Step backend with
  custom model loading, dtype policy, MPS fallbacks, null lyric handling, VAE decode tiling, and
  WAV encoding.
- `src/abstractmusic/backends/diffusers_audio.py` is generic, but it does not understand
  `AceStepPipeline`'s `audio_duration`, `lyrics`, or `vocal_language` arguments.
- `tests/` contains useful fake-backed unit coverage, and
  `python -m pytest -q -m "not integration"` passed on 2026-05-15.
- The repo root contains generated `.wav` files and Python bytecode exists under `src/` and
  `tests/`; these are hygiene failures, not source.
- `docs/` lacks the required documentation baseline except for one proposed backlog item.

## Problem

The project can pass unit tests without proving that it can generate usable music with real models.
The current package also mixes framework contracts with heavyweight model runtime dependencies and
does not yet provide a clear provider/capability abstraction.

## What we want to do

Preserve this assessment as the roadmap entry point, then implement the planned backlog in priority
order.

## Why

Future implementation agents need a clear explanation of why the work is ordered this way. Without
that, it is easy to keep patching the large ACE-Step backend while missing the abstraction, docs,
packaging, and validation issues that make the project feel broken.

## Requirements

- Keep a unified music generation API above model-specific backends.
- Prioritize real, permissive, open-source music generation over demos that only pass with mocks.
- Keep dependencies minimal and lazy.
- Do not make non-commercial or license-unclear models defaults.
- Do not silently ignore request fields such as lyrics, duration, seed, negative prompt, or
  guidance scale.

## Suggested implementation

Use this item as the first completion target after the backlog is accepted. Its completion report
should point to the created backlog items and any immediate code/docs findings discovered while
implementing them.

## Scope

- Record current project status.
- Record model research conclusions.
- Define the implementation order.

## Non-goals

- Do not implement provider changes in this item.
- Do not remove artifacts or change packaging here unless completing another backlog item.

## Dependencies and related tasks

- `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- `docs/backlog/planned/010_repo_hygiene_docs_and_packaging.md`
- `docs/backlog/planned/020_music_abstraction_and_capability_registry.md`
- `docs/backlog/planned/030_acestep_diffusers_xl_provider.md`
- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- `docs/backlog/planned/050_dependency_profiles_and_optional_providers.md`

## Expected outcomes

- The planned backlog gives a coherent path from current state to a working AbstractMusic package.
- The model/provider choices are justified from current evidence, not guesses.

## Validation

- Read the planned items and verify each one is standalone.
- Run `python -m pytest -q -m "not integration"` after any code edits made while completing
  related tasks.

## Progress checklist

- [ ] Review this assessment against the current code.
- [ ] Update stale statements if code has changed.
- [ ] Add completion report when the roadmap is accepted or superseded.

## Guidance for the implementing agent

Treat this item as orientation, not authority. If the code has moved, update the backlog before
coding.

## Completion report

### Summary

Completed the critical assessment and turned it into an actionable backlog. The assessment
identified the core risks: mock-only validation, tracked generated artifacts, missing repo hygiene,
overly heavy base dependencies, loose dependency bounds around vendored ACE-Step code, and the need
for a real provider/model abstraction.

### Files and docs touched

- Added `docs/backlog/overview.md`
- Added planned provider, hygiene, validation, dependency, ACE-Step, and HeartMuLa backlog items
- Added `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`

### Validation

- `python -m pytest -q -m "not integration"` passed after implementation work.
- `python -m build && python -m twine check dist/*` passed after packaging fixes.

### Residual risks

Real model generation still needs the planned opt-in smoke matrix. Dependency profile splitting is
still planned work.
