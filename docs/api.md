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

### Styles (instruments, vocal traits, mix notes)

You can pass lightweight provider-neutral “style tags” that describe instruments, vocalist traits, or
production notes.

When present:

- `elevenlabs` uses them as a provider-neutral `composition_plan`.
- other engines fold `positive_styles` into the prompt text as additional tags.

```python
asset = music.generate_audio(
    "smoky jazz with a clear hook",
    duration_s=25,
    planning=True,
    positive_styles=["saxophone", "brushed drums", "female vocalist"],
    negative_styles=["no autotune"],
)
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
`available_providers(task=...)`, `provider_details(task=...)`,
`list_models(task=..., provider=...)`, `list_provider_models(...)`, `list_operations(task=...)`,
and `capability_catalog(task=...)`.
They use the packaged model capability registry and only surface providers/models that can run right
now. Provider filters are backend-oriented ids such as `acemusic`, `elevenlabs`, `acestep`,
`stable-audio`, `stable-audio-3`, and `diffusers`.

Discovery never imports a model runtime and never loads weights. Availability is answered by two
cheap probes in `abstractmusic.availability`:

- **Local providers** — the optional runtime must be installed, and at least one of the provider's
  models must already have weights in the Hugging Face cache. Cache-root resolution mirrors
  `huggingface_hub` (`HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then
  `~/.cache/huggingface/hub`), so discovery looks where the loader will look. Each model record
  carries `metadata.cached`.
- **Remote providers** — a provider with an API key configured is probed at its own read-only
  endpoint. All remote providers are probed concurrently in one round, bounded by a 5s deadline
  (`abstractmusic.availability.REMOTE_PROBE_TIMEOUT_S`). A provider that rejects the key or does not
  answer is not reported as available. Results are reused for a short window, and an unresponsive
  provider is never probed twice concurrently, so a `capability_catalog(...)` call pays that
  deadline at most once.

`available_providers(...)` answers "what can I run"; `provider_details(...)` answers "what else is
there, and what is missing". It returns every known provider with a `usable` boolean, a `status`
(`available`, `unauthorized`, `unavailable`, `unreachable`, `not-configured`, `not-installed`,
`no-local-weights`), `metadata.reason`, the provider's full model catalog in `metadata.models`, and
the subset already downloaded in `metadata.cached_models`. Use it whenever `available_providers(...)`
comes back short, and to show users what they could install or download.

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
spec = registry.get("ACE-Step/acestep-v15-xl-turbo-diffusers")
```

The registry is metadata only. It must not silently change the configured provider or model.

## Built-In Backend Kinds

- `acemusic`: default lightweight remote backend for the ACE Music hosted API. It requires a remote
  API key and can request WAV, MP3, or FLAC.
- `elevenlabs`: lightweight remote backend for ElevenLabs Music only. It requires
  `ELEVENLABS_API_KEY`, calls `/v1/music` and `/v1/music/plan`, and can request WAV or MP3. Voice
  and text-to-speech routes are intentionally not exposed here.
- `acestep`: local ACE-Step adapter for `AceStepPipeline`. Defaults to the official
  `ACE-Step/acestep-v15-xl-turbo-diffusers` checkpoint. Other trained AceStepPipeline-compatible
  checkpoints can be selected explicitly with `model_id`, but the packaged discovery catalog only
  surfaces reviewed entries.
- `diffusers`: generic Diffusers audio backend for compatible audio pipelines.
- `musicgen`: Transformers MusicGen adapter for `facebook/musicgen-small` (non-commercial).
- `stable-audio`: vendored Stable Audio Open Small adapter for
  `stabilityai/stable-audio-open-small` (gated, non-default, short clips).
- `stable-audio-3`: internal AbstractMusic runtime for `stabilityai/stable-audio-3-small-music`
  and tracked Medium support. Uses Hugging Face weights/configs only; it does not import the
  upstream `stable_audio_3` package.
