# Music Models

AbstractMusic keeps known model metadata in
`src/abstractmusic/assets/music_model_capabilities.json`.

This list is used for capability discovery and planning. It is not a runtime router and must not
silently change the configured model.

Runtime model selection accepts Hugging Face repo ids only. AbstractMusic does not accept local
checkpoint directory paths or custom Hugging Face cache directories.

## Current Reviewed Models

- `ACE-Step/acestep-v15-xl-turbo-diffusers`: recommended through the default `acestep` /
  `acestep-diffusers` backend, MIT, text-to-music with lyrics. This path uses Diffusers
  AceStepPipeline, Hugging Face checkpoint files, and package-owned orchestration without an
  external ACE-Step source tree or package. On Apple MPS, AbstractMusic avoids fp16 denoising
  overflow by preferring MPS bfloat16 when supported and MPS float32 otherwise; CPU float32 remains
  the final fallback if MPS returns non-finite audio.
- `ACE-Step/Ace-Step1.5`: explicit `acestep-v15` backend, MIT, text-to-music with lyrics. This
  path uses vendored ACE-Step model code and package-owned orchestration without an external
  ACE-Step source tree or package, but it is quality-limited after repeated-loop validation
  failures.
  The experimental 5Hz audio-code planner is opt-in because coarse code hints can imprint
  repetitive artifacts. The default turbo DiT does not use CFG, so
  `guidance_scale` is treated as unsupported unless a non-turbo DiT is explicitly configured.
- `facebook/musicgen-small`: 300M text-to-music model through Transformers, CC BY-NC 4.0. This is
  configured as the optional `musicgen` backend and remains the best small validation candidate
  because its inference path is straightforward and model family is established, but the weights
  are non-commercial.
- `stabilityai/stable-audio-open-small`: 341M gated text-to-audio model, Stability AI Community
  License, short 11-second clips. It is configured as the optional `stable-audio` backend and is
  interesting for Apple/Arm-friendly short clips and sound effects, but not a strong default music
  candidate. Hugging Face access approval is required before weights can be downloaded.
- `ACE-Step/acestep-v15-xl-turbo`: raw XL Turbo DiT checkpoint, MIT, heavy advanced variant.
- `ACE-Step/acestep-v15-xl-sft`: raw XL SFT checkpoint, MIT, heavy quality variant with guidance.
- `HeartMuLa/HeartMuLa-oss-3B-happy-new-year`: Apache-2.0, lyrics and tag conditioned music
  generation with HeartCodec.
- `m-a-p/YuE-s1-7B-anneal-en-cot`: Apache-2.0 YuE Stage-1 music model. It is not an
  end-to-end audio checkpoint by itself; a provider needs the full Stage-2 and codec/upsampler
  pipeline before it can be considered working music generation.
- `LH-Tech-AI/TinyMozart_v2_85M`: small unconditional MIDI piano generator, license not declared
  in reviewed metadata.
- `Dalision/Omni2Sound`: CC BY-NC 4.0 multimodal audio generation, not suitable as a default
  commercial-capable music provider.

## Precision Rule

Prefer official 8-bit artifacts when available. If none are available, prefer official 16-bit
artifacts. The reviewed models currently do not expose official 8-bit checkpoints in their Hugging
Face metadata.

On Apple hardware, use PyTorch MPS for standalone providers when supported and keep dtype/fallback
events explicit in logs/metadata. CPU is a fallback path, not the default acceleration strategy.
