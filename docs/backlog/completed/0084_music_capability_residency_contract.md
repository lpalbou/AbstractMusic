# Completed: Music Capability Residency Contract

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Priority: P1

## Context

AbstractMusic already exposes Core-facing discovery and generation, but not residency:

- `available_providers(...)`
- `list_models(...)`
- `list_provider_models(...)`
- `t2m(...)`

It does not expose:

- `load_resident_model(...)`
- `list_loaded_models(...)`
- `list_resident_models(...)`
- `unload_resident_model(...)`

That means AbstractCore cannot truthfully load, list, or unload local music engines. Discovery must
not be used as a substitute for loaded state.

## Minimal proposal

Add the same optional residency surface used by the other capability plugins:

```python
def load_resident_model(request): ...
def list_loaded_models(filters=None): ...
def list_resident_models(filters=None): ...
def unload_resident_model(request): ...
```

Use existing backend behavior:

- local backends that already keep a pipeline/model in memory may implement preload/list/unload;
- remote ACE Music and ElevenLabs remain stateless and should not appear as loaded;
- listing returns only real in-process loaded runtimes;
- generation should reuse a loaded backend instance when the provider/model match.

AbstractCore also needs to route `music_generation` through the generic capability-residency path
once AbstractMusic exposes these methods.

## Non-goals

- No new provider catalog design.
- No new local music backend.
- No claim that `loaded=true` means first generation is fully hot.
- No loading during discovery.
- No API keys, prompts, lyrics, or generated audio in residency records.

## Success criteria

- Loading a local music backend makes `list_loaded_models()` show one real loaded record.
- Unloading that backend removes the loaded record and releases the local runtime.
- Remote providers do not produce fake loaded records.
- Core `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload` can use the same
  contract as voice and vision.

## Test plan

- Unit-test fake local backend load/list/unload.
- Unit-test remote providers stay stateless.
- Unit-test discovery does not load a model.
- Add one Core contract test once Core dispatches `music_generation`.

## Completion report

2026-05-21:

- Added `load_resident_model(...)`, `list_loaded_models(...)`, `list_resident_models(...)`, and
  `unload_resident_model(...)` to the AbstractCore capability objects so Core can warm, list, and
  unload local music runtimes truthfully.
- Implemented best-effort `preload()` / `unload()` hooks on local backends so residency calls can
  reuse or release in-process pipelines without loading during discovery.
- Added unit tests proving local load/list/unload works and remote providers remain stateless.

Touched:

- `src/abstractmusic/integrations/abstractcore_plugin.py`
- `src/abstractmusic/backends/acestep_diffusers.py`
- `src/abstractmusic/backends/acestep_v15.py`
- `src/abstractmusic/backends/diffusers_audio.py`
- `src/abstractmusic/backends/musicgen.py`
- `src/abstractmusic/backends/stable_audio.py`
- `src/abstractmusic/backends/stable_audio_3.py`
- `tests/test_abstractcore_plugin_local.py`

Validation:

- `python -m pytest -q`
