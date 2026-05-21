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

`t2m(...)` is a convenience method that returns audio bytes directly. WAV is the baseline format;
remote backends may support additional formats.

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

In AbstractCore plugin mode, AbstractMusic also accepts a host-supplied text service structurally
when the owner context/config exposes `generate_text(...)` or `generate_structured(...)`. This keeps
LLM planning injectable without adding an AbstractCore dependency or passing raw provider objects
into AbstractMusic.

The plugin capability object exposes AbstractCore-friendly discovery methods:
`available_providers(task=...)`, `list_models(task=..., provider=...)`,
`list_provider_models(...)`, `list_operations(task=...)`, and `capability_catalog(task=...)`.
They use the packaged model capability registry and do not load model weights.

When running under AbstractCore, the capability object also exposes an optional residency surface
for local backends:

- `load_resident_model(request)`
- `list_loaded_models(filters=None)`
- `list_resident_models(filters=None)`
- `unload_resident_model(request)`

Remote providers remain stateless and do not show up as loaded. Local backends implement best-effort
`preload()` / `unload()` so AbstractCore can warm and release in-process music engines without
abusing discovery as a proxy for residency.

Planners that understand song structure can return `composition_plan` in addition to prompt/lyrics.
Backends that support structured plans, such as `elevenlabs`, translate that provider-neutral plan
into their native request format. Backends that do not support plans continue using the compiled
prompt/lyrics fields.

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
- `composition_plan`
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

- `acemusic`: default lightweight remote backend for the ACE Music hosted API. It requires a remote
  API key and can request WAV, MP3, or FLAC.
- `elevenlabs`: lightweight remote backend for ElevenLabs Music only. It requires
  `ELEVENLABS_API_KEY`, calls `/v1/music` and `/v1/music/plan`, and can request WAV or MP3. Voice
  and text-to-speech routes are intentionally not exposed here.
- `acestep`: local ACE-Step Diffusers XL Turbo backend alias.
- `acestep-diffusers`: local ACE-Step Diffusers adapter for `AceStepPipeline`. Defaults to the
  official `ACE-Step/acestep-v15-xl-turbo-diffusers` checkpoint, but also supports compatible
  Diffusers conversions when `model_id` is set (for example, community conversions hosted on
  Hugging Face).
- `acestep-v15`: explicit quality-limited ACE-Step v1.5 backend, using vendored model code and
  package-owned orchestration.
- `diffusers`: generic Diffusers audio backend for compatible audio pipelines.
- `musicgen`: Transformers MusicGen adapter for `facebook/musicgen-small` (non-commercial).
- `stable-audio`: stable-audio-tools adapter for `stabilityai/stable-audio-open-small`
  (gated, non-default, short clips).
- `stable-audio-3`: internal AbstractMusic runtime for `stabilityai/stable-audio-3-small-music`
  and tracked Medium support. Uses Hugging Face weights/configs only; it does not import the
  upstream `stable_audio_3` package.
