# Music Models

AbstractMusic keeps known model metadata in
`src/abstractmusic/assets/music_model_capabilities.json`.

This list is used for capability discovery and planning. It is not a runtime router and must not
silently change the configured model.

Local runtime model selection accepts Hugging Face repo ids only. AbstractMusic does not accept
local checkpoint directory paths or custom Hugging Face cache directories. Remote providers use
provider-specific model names when exposed by the service.

## Current Reviewed Models

- `acemusic/ace-step-api`: recommended lightweight remote backend through `acemusic`, text-to-music
  with optional lyrics. It uses the configured ACE Music hosted API and requires an API key;
  licensing and commercial terms are provider-side.
- `elevenlabs/music_v1`: recommended lightweight remote backend through `elevenlabs`, text-to-music
  with optional lyrics and composition plans. It calls only ElevenLabs Music endpoints; voice and
  text-to-speech belong in AbstractVoice. Live use may require a paid Music-enabled ElevenLabs
  account tier.
- `ACE-Step/acestep-v15-xl-turbo-diffusers`: recommended through the local `acestep` backend,
  MIT, text-to-music with lyrics. This path uses Diffusers
  AceStepPipeline, Hugging Face checkpoint files, and package-owned orchestration without an
  external ACE-Step source tree or package. On Apple MPS, AbstractMusic avoids fp16 denoising
  overflow by preferring MPS bfloat16 when supported and MPS float32 otherwise; CPU float32 remains
  the final fallback if MPS returns non-finite audio.
  The same backend can also target other trained AceStepPipeline-compatible checkpoints by passing
  `--model-id`, but discovery only surfaces reviewed models.
- `facebook/musicgen-small`: 300M text-to-music model through Transformers, CC BY-NC 4.0. This is
  configured as the optional `musicgen` backend and remains the best small validation candidate
  because its inference path is straightforward and model family is established, but the weights
  are non-commercial.
- `stabilityai/stable-audio-open-small`: 341M gated text-to-audio model, Stability AI Community
  License, short 11-second clips. It is configured as the optional `stable-audio` backend and is
  interesting for Apple/Arm-friendly short clips and sound effects, but not a strong default music
  candidate. Hugging Face access approval is required before weights can be downloaded.
- `stabilityai/stable-audio-3-small-music`: gated Stable Audio 3 Small Music checkpoint, Stability
  AI Community License plus text-encoder terms, 44.1 kHz stereo, up to 120 seconds. It is exposed
  through `--backend stable-audio-3` using AbstractMusic-owned internal runtime code and Hugging
  Face weights/configs only. Initial scope is text-to-music; LoRA, inpainting, continuation,
  audio-to-audio, CoreML/TFLite, and TensorRT are deferred. It has passed focused 30-second and
  120-second local validation runs at 16 steps, but it is not recommended until broader
  prompt/seed and GPU validation are complete.
- `stabilityai/stable-audio-3-small-sfx`: gated Stable Audio 3 Small SFX checkpoint. It is exposed
  through the same `stable-audio-3` backend and is intended for longer text-to-audio sound effects.
  It is not recommended until it has dedicated SFX validation coverage.
- `stabilityai/stable-audio-3-medium`: gated Stable Audio 3 Medium checkpoint, tracked behind
  `stable-audio-3` but not recommended until Small Music has broader validation. It is GPU-oriented
  and heavier than the Small model.
- `HeartMuLa/HeartMuLa-oss-3B-happy-new-year`: Apache-2.0, lyrics and tag conditioned music
  generation with HeartCodec.
- `m-a-p/YuE-s1-7B-anneal-en-cot`: Apache-2.0 YuE Stage-1 music model. It is not an
  end-to-end audio checkpoint by itself; a provider needs the full Stage-2 and codec/upsampler
  pipeline before it can be considered working music generation.
- `LH-Tech-AI/TinyMozart_v2_85M`: small unconditional MIDI piano generator, license not declared
  in reviewed metadata.
- `Dalision/Omni2Sound`: CC BY-NC 4.0 multimodal audio generation, not suitable as a default
  commercial-capable music provider.

## ACE-Step Scope

The public ACE-Step contract is one backend, `acestep`, plus a model id. The reviewed discovery
catalog currently surfaces these official ACE-Step checkpoints:

- `ACE-Step/Ace-Step1.5`
- `ACE-Step/acestep-v15-base`
- `ACE-Step/acestep-v15-sft`
- `ACE-Step/acestep-v15-xl-turbo-diffusers`

The validated default remains `ACE-Step/acestep-v15-xl-turbo-diffusers`.

The 5 Hz language-model checkpoints such as `ACE-Step/acestep-5Hz-lm-0.6B` are not surfaced in the
music picker because they are planner/code models, not direct end-to-end music generation models.

If you want to try other trained ACE-Step checkpoints, pass the Hugging Face repo id explicitly and
validate the result locally before treating it as supported.

## Precision Rule

Prefer official 8-bit artifacts when available. If none are available, prefer official 16-bit
artifacts. The reviewed models currently do not expose official 8-bit checkpoints in their Hugging
Face metadata.

On Apple hardware, use PyTorch MPS for standalone providers when supported and keep dtype/fallback
events explicit in logs/metadata. CPU is a fallback path, not the default acceleration strategy.
