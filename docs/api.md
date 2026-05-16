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

- `acestep`: custom ACE-Step v1.5 backend for `ACE-Step/Ace-Step1.5`.
- `acestep-official`: upstream ACE-Step v1.5 handler plus 5Hz LM planner for
  `ACE-Step/Ace-Step1.5`; recommended ACE-Step path.
- `acestep-diffusers`: ACE-Step Diffusers adapter for
  `ACE-Step/acestep-v15-xl-turbo-diffusers`.
- `diffusers`: generic Diffusers audio backend for compatible audio pipelines.
- `musicgen`: Transformers MusicGen adapter for `facebook/musicgen-small` (non-commercial).
- `stable-audio`: stable-audio-tools adapter for `stabilityai/stable-audio-open-small`
  (gated, non-default, short clips).
