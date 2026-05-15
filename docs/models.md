# Music Models

AbstractMusic keeps known model metadata in
`src/abstractmusic/assets/music_model_capabilities.json`.

This list is used for capability discovery and planning. It is not a runtime router and must not
silently change the configured model.

## Current Reviewed Models

- `ACE-Step/Ace-Step1.5`: recommended through `acestep-official`, MIT, text-to-music with lyrics.
  This path wraps the upstream ACE-Step handler and 5Hz LM planner, and defaults to the bundled
  `acestep-5Hz-lm-1.7B` LM. Short instrumental generations remain quality-risky: recent 10-second
  smokes produced harmonic audio but were still highly repetitive. The older packaged custom
  backend remains non-recommended because a 3-second Apple MPS run produced valid PCM but sounded
  and measured like fast rotor-like audio.
- `ACE-Step/acestep-v15-xl-turbo-diffusers`: preferred next ACE-Step XL provider candidate, MIT,
  Diffusers `AceStepPipeline`, text-to-music with lyrics; adapter implemented, real model
  validation pending.
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

On Apple hardware, use an official MLX path when one exists for a provider; otherwise prefer
PyTorch MPS. CPU is a fallback path, not the default acceleration strategy.
