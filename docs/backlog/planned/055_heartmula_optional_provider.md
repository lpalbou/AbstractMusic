# Planned: HeartMuLa Optional Provider

## Metadata
- Created: 2026-05-15
- Status: Planned
- Completed: N/A
- Priority: P2

## Context

HeartMuLa publishes `HeartMuLa/HeartMuLa-oss-3B-happy-new-year`, which the Hugging Face model card
describes as an open-source text-to-audio/music model with strong lyrics controllability and music
quality. The official `HeartMuLa/heartlib` GitHub repo documents local generation conditioned on
lyrics and tags with the paired HeartCodec checkpoint.

## Current code reality

- AbstractMusic has no HeartMuLa backend.
- `src/abstractmusic/assets/music_model_capabilities.json` includes HeartMuLa metadata as a
  planned research provider.
- The current unified request type has first-class `lyrics`, `vocal_language`, `duration_s`,
  `guidance_scale`, `format`, and `extra` fields. HeartMuLa also needs tags.
- HeartMuLa's official docs recommend Python 3.10, downloading `HeartMuLaGen`,
  `HeartMuLa-oss-3B-happy-new-year`, and `HeartCodec-oss-20260123`, then running
  `examples/run_music_generation.py`.
- The Hugging Face model files are sharded safetensors and the model card reports Apache-2.0
  licensing. The reviewed model page reported F32 tensor type; official runtime docs default
  HeartMuLa to bf16 and HeartCodec to fp32.

## Problem

HeartMuLa is a credible open-source music generator, but it is not a simple single-checkpoint
Diffusers pipeline. A naive integration would pull a heavy runtime stack and paired codec into the
base package.

## What we want to do

Evaluate and, if practical, implement HeartMuLa as an optional provider behind a dedicated extra
and explicit backend id.

## Why

HeartMuLa could complement ACE-Step because it is Apache-2.0, explicitly music-focused, and
lyrics/tags conditioned. It may be valuable for users who need controllable song generation.

## Requirements

- Keep HeartMuLa out of base dependencies.
- Use only official sources for model artifacts and runtime code.
- Prefer official 8-bit artifacts if HeartMuLa publishes them. If none exist, use official bf16
  for HeartMuLa and fp32 for HeartCodec as documented by upstream.
- Add `tags` support in a typed way if it becomes a common music-generation field; otherwise pass
  it through an explicit HeartMuLa config/extra path.
- Support `lyrics`, `duration_s`/`max_audio_length_ms`, `guidance_scale`/`cfg_scale`, seed or
  sampling controls if upstream exposes them, and output format handling.
- Avoid adding external codecs to base. If upstream only writes MP3, either keep MP3 as provider
  output or add a minimal, explicit conversion path under the HeartMuLa extra.
- Add real smoke validation gated by env vars because the model and codec are large.
- Document hardware, memory, expected runtime, and lazy-load behavior.

## Suggested implementation

Start with a wrapper around the official `heartlib` runtime or CLI in an optional backend. Only
recode internal glue where it reduces dependency surface without copying a large upstream project.
If official APIs are unstable, keep the provider experimental.

## Scope

- Provider feasibility spike.
- Optional dependency profile.
- Backend/config implementation if feasible.
- Registry/docs/tests updates.

## Non-goals

- Do not make HeartMuLa the default provider in this task.
- Do not vendor large upstream code unless there is no stable package/API and the license review is
  complete.
- Do not use unofficial quantized checkpoints as defaults.

## Dependencies and related tasks

- `docs/backlog/planned/020_music_abstraction_and_capability_registry.md`
- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- `docs/backlog/completed/050_dependency_profiles_and_optional_providers.md`
- Sources:
  - `https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year`
  - `https://github.com/HeartMuLa/heartlib`

## Expected outcomes

- Clear decision on whether HeartMuLa is a supported optional provider or research-only.
- If implemented, users can generate music through `MusicManager` with the same public abstraction.
- Heavy HeartMuLa/HeartCodec dependencies remain explicit.

## Validation

- `python -m pytest -q -m "not integration"`
- Fake-backed provider tests for argument mapping.
- Opt-in real smoke test using official checkpoints and a short lyrics/tags input.
- WAV/MP3 artifact validation and no generated media tracked in git.

## Progress checklist

- [ ] Inspect official `heartlib` runtime API and dependency footprint.
- [ ] Decide wrapper vs minimal internal implementation.
- [ ] Add optional extra and backend config if feasible.
- [ ] Add tests and docs.
- [ ] Run real smoke validation when hardware/checkpoints permit.

## Guidance for the implementing agent

Keep the provider optional and explicit. The main risk is not whether HeartMuLa can generate music;
the main risk is making its heavy paired-codec runtime part of AbstractMusic's base install.
