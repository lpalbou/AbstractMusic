# Completed: Truthful Stable Audio Capability Registration And Music Routing

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Priority: P0

## Context

AbstractMusic already has a backend implementation for
`stabilityai/stable-audio-open-small`, and the model registry plus CLI aliases advertise it as an
optional local engine. Higher layers now surface music provider/model discovery, so that catalog
must correspond to a real capability backend that Core can select.

Live reproduction on 2026-05-21 showed that this is not currently true. A request asking for
`stabilityai/stable-audio-open-small` through Gateway/Core completed, but the actual backend used
was `abstractmusic:acemusic`. This is partly a Core selection bug, but it also exposed an
AbstractMusic contract gap: the AbstractCore plugin does not register a Stable Audio Open Small
capability backend at all, even though the package advertises the model and CLI route.

## Current Code Reality

- `src/abstractmusic/backends/stable_audio.py` implements `StableAudioBackend` with
  `model_id="stabilityai/stable-audio-open-small"`, `max_duration_s=11`, and request-level
  `duration_s` handling.
- `src/abstractmusic/integrations/abstractcore_plugin.py` registers capability backends for
  `abstractmusic:acemusic`, `abstractmusic:elevenlabs-music`,
  `abstractmusic:acestep-diffusers`, `abstractmusic:acestep-v15`,
  `abstractmusic:stable-audio-3`, and `abstractmusic:diffusers`.
  It does not register a backend for `abstractmusic:stable-audio`.
- `src/abstractmusic/assets/music_model_capabilities.json` advertises
  `stabilityai/stable-audio-open-small` with `backend_kinds=["stable-audio"]`,
  `dependency_extra="stable-audio"`, and local/gated notes.
- `src/abstractmusic/cli.py` already maps CLI aliases such as `stable-audio` and
  `stable-audio-open-small` to the Stable Audio Open Small backend.
- `tests/test_stable_audio_backend.py` covers backend behavior directly.
- `tests/test_abstractcore_plugin_local.py` exercises plugin registration and discovery, including
  `abstractmusic:stable-audio-3`, but there is no equivalent coverage proving that a Stable Audio
  Open Small capability backend is registered and selectable through the AbstractCore plugin.
- `docs/backlog/deprecated/0075_stable_audio_open_small_validation.md` tracked the original Open Small
  validation intent.
  validation, but not the missing capability registration/routing truth that higher layers need.

## Problem

AbstractMusic currently tells the ecosystem two different stories:

- the backend/model registry and CLI say Stable Audio Open Small exists as a selectable engine;
- the AbstractCore plugin does not expose a matching capability backend.

That makes provider/model discovery optimistic in a way higher layers cannot satisfy. It also
creates room for wrong backend selection below Gateway and Runtime.

## What We Want To Do

Make Stable Audio Open Small truthful at the capability boundary.

The package should either:

- expose a real `abstractmusic:stable-audio` capability backend through the AbstractCore plugin, or
- stop advertising Stable Audio Open Small as a capability-selectable backend until that support
  exists.

This item assumes the intended fix is to expose the backend cleanly.

## Why

Discovery is part of the public contract now. If AbstractMusic advertises a local model/backend,
AbstractCore, Runtime, and Gateway need a real backend to target. Otherwise every higher layer
looks wrong even when it simply follows the catalog.

## Requirements

- Register a dedicated AbstractCore plugin backend for Stable Audio Open Small, for example
  `abstractmusic:stable-audio`.
- Keep Stable Audio Open Small separate from Stable Audio 3. The backends, model limits, and
  dependency extras are different and must not be conflated.
- Ensure the plugin-level discovery/catalog surface can represent Stable Audio Open Small
  truthfully, including installed/configured/gated state.
- Keep dependency loading lazy and optional. The base package must remain free of Stable Audio
  runtime dependencies.
- Make backend/model naming consistent across:
  - plugin registration
  - model capability registry metadata
  - CLI alias vocabulary
  - Core-facing backend selector expectations
- Add plugin-level regression tests proving the backend is registered and that discovery/catalog
  records for `stabilityai/stable-audio-open-small` are routed through the matching capability
  backend.

## Suggested Implementation

1. Add a dedicated `_AbstractMusicStableAudioCapability` in
   `src/abstractmusic/integrations/abstractcore_plugin.py` that wraps
   `StableAudioBackend` and exposes the same minimal `t2m(...)` contract as the other capability
   backends.
2. Register it with a concrete backend id such as `abstractmusic:stable-audio`, a clear config
   hint, and a priority that does not override unrelated local backends by accident.
3. Keep `stabilityai/stable-audio-open-small` in the model capability registry, but ensure the
   discovery records tie that model to the new registered backend instead of leaving it as a
   backend kind with no matching capability.
4. Add plugin tests covering registration, discovery, and selection truth for Stable Audio Open
   Small.
5. Leave quality validation and recommendation decisions to a future provider-validation task
   after the capability boundary is fixed.

## Scope

- AbstractCore integration module registration for Stable Audio Open Small.
- Discovery/catalog truth for that backend/model pair.
- Tests and docs needed so higher layers can trust the catalog.

## Non-Goals

- Do not promote Stable Audio Open Small to the default music backend.
- Do not collapse Stable Audio Open Small and Stable Audio 3 into one ambiguous backend.
- Do not mark the model recommended just because registration exists.
- Do not solve Core's request-scoped selector bug here; that is tracked separately in
  `../abstractcore/docs/backlog/planned/0799_request_scoped_music_backend_selection_and_truthful_reporting.md`.

## Dependencies And Related Tasks

- `docs/backlog/deprecated/0075_stable_audio_open_small_validation.md`
- `../abstractcore/docs/backlog/planned/0799_request_scoped_music_backend_selection_and_truthful_reporting.md`
- `src/abstractmusic/backends/stable_audio.py`
- `src/abstractmusic/integrations/abstractcore_plugin.py`
- `src/abstractmusic/assets/music_model_capabilities.json`
- `tests/test_stable_audio_backend.py`
- `tests/test_abstractcore_plugin_local.py`

## Expected Outcomes

- AbstractMusic's model registry, CLI aliases, and AbstractCore plugin all agree that Stable Audio
  Open Small is a distinct optional backend.
- Higher layers can request the backend truthfully instead of depending on catalog fiction.
- The follow-up validation item for Stable Audio Open Small becomes a real provider-validation task
  instead of a half-routing, half-quality investigation.

## Validation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_stable_audio_backend.py tests/test_abstractcore_plugin_local.py tests/test_model_capabilities.py tests/test_cli_smoke.py`
- Add a plugin-level regression asserting the registered backend ids include
  `abstractmusic:stable-audio` and that the discovery/catalog path can surface
  `stabilityai/stable-audio-open-small` through that backend without claiming recommendation or
  ungated availability.

## Progress Checklist

- [x] Add a Stable Audio Open Small capability backend to the AbstractCore integration plugin.
- [x] Align model registry/catalog data with the registered backend id.
- [x] Add plugin regression tests for registration and discovery truth.

## Completion report

2026-05-21:

- Implemented a dedicated `abstractmusic:stable-audio` AbstractCore plugin backend that wraps
  `StableAudioBackend` (Stable Audio Open Small via AbstractMusic-vendored stable-audio-tools model code).
- Aligned AbstractCore-facing model discovery so catalog entries route to the correct registered
  `backend_id` based on `backend_kinds`, avoiding "advertised but not selectable" backends.
- Added plugin registration coverage proving `abstractmusic:stable-audio` is registered.

Touched:

- `src/abstractmusic/integrations/abstractcore_plugin.py`
- `tests/test_abstractcore_plugin_local.py`

Validation:

- `python -m pytest -q`

Notes:

- This fixes AbstractMusic-side capability truth, but does not address Core's request-scoped
  backend selection/reporting bug; that remains tracked in AbstractCore backlog.

## Guidance For The Implementing Agent

Do not paper over this by deleting the model from the catalog unless the package is intentionally
dropping Stable Audio Open Small support. The better fix is to make the advertised backend real and
keep the catalog honest about gating and validation state.
