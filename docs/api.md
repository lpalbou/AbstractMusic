# API

## MusicManager

```python
from abstractmusic import MusicManager
```

`MusicManager` delegates generation to a configured backend.

```python
asset = music.generate_audio(
    "cinematic orchestral intro",
    lyrics="[Verse]\n...",
    duration_s=10.0,
    seed=123,
    format="wav",
)
```

`generate_audio(...)` returns a `GeneratedAsset` unless a store is configured, in which case it can
return an artifact reference.

`t2m(...)` is a convenience method that returns WAV bytes directly.

## Text Planning

AbstractMusic exposes a small planner boundary so host applications can improve prompt rewriting
without changing backend implementations:

```python
def planner(request_dict):
    return {
        "prompt": "cinematic orchestral intro with a brass theme and clear build",
        "lyrics": "[Instrumental]",
        "bpm": 96,
        "keyscale": "D minor",
        "planner_backend": "host-llm",
        "generated_fields": ["prompt", "lyrics", "bpm", "keyscale"],
    }

music = MusicManager(backend=backend, text_planner=planner, text_planner_mode="auto")
asset = music.generate_audio("heroic fantasy", duration_s=30, planning=True)
```

Planner providers can be callables accepting `request_dict`, objects with
`plan_music_text(request_dict)`, or objects with `create_plan(MusicPlanningRequest)`. `auto` uses
the provider when present and falls back to the deterministic local planner. `required` raises on
provider failure. `off` preserves raw user text.

## Request Fields

Core fields:

- `prompt`
- `lyrics`
- `vocal_language`
- `negative_prompt`
- `duration_s`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `format`
- `sample_rate`
- `extra`

Backends must raise clear errors for unsupported fields when support is known.

## Model Registry

```python
from abstractmusic.model_capabilities import MusicModelCapabilitiesRegistry

registry = MusicModelCapabilitiesRegistry()
spec = registry.get("ACE-Step/Ace-Step1.5")
```

The registry is metadata only. It must not silently change the configured provider or model.

## Built-In Backend Kinds

- `acestep`: default ACE-Step Diffusers XL Turbo backend.
- `acestep-diffusers`: explicit name for the default ACE-Step Diffusers adapter for
  `ACE-Step/acestep-v15-xl-turbo-diffusers`.
- `acestep-v15`: explicit quality-limited ACE-Step v1.5 backend, using vendored model code and
  package-owned orchestration.
- `diffusers`: generic Diffusers audio backend for compatible audio pipelines.
- `musicgen`: Transformers MusicGen adapter for `facebook/musicgen-small` (non-commercial).
- `stable-audio`: stable-audio-tools adapter for `stabilityai/stable-audio-open-small`
  (gated, non-default, short clips).
