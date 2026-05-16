# Completed: Official ACE-Step V1.5 MLX Provider

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-15
- Priority: P0

## Context

The custom ACE-Step v1.5 backend can produce a valid WAV, but the first Apple MPS smoke was
rejected as rotor-like rather than music. Inspection of the official ACE-Step v1.5 repo shows that
the full path uses the `acestep-5Hz-lm-1.7B` language model to generate semantic audio codes before
DiT diffusion. The custom backend currently bypasses that phase.

## Problem

AbstractMusic still has no validated open-source path that generates usable music through the
unified abstraction. A PCM-only smoke is insufficient; the provider must pass a music-likeness gate
with harmonic structure and no dominant fast repetitive envelope.

## What we want to do

Add an official ACE-Step v1.5 provider that wraps the upstream handler/API with minimal local glue,
preferring MLX on Apple Silicon and GPU-native paths on CUDA/XPU. Keep it behind an explicit extra
and preserve the unified `MusicBackend` request/result contract.

## Why

The official pipeline already encodes the music-planning behavior that the custom path appears to
be missing. Reusing only the necessary upstream runtime pieces is more likely to produce music than
continuing to patch a partial DiT-only implementation.

## Requirements

- Use official Hugging Face model sources only.
- Prefer official 8-bit artifacts if they appear; otherwise use official 16-bit/model-default
  precision. Do not invent local quantization as the default.
- On Apple hardware, initialize the 5Hz LM with MLX first; use PyTorch MPS/MLX DiT support where
  upstream supports it; CPU is fallback only.
- Generate at least one 10-second instrumental sample through AbstractMusic's public abstraction.
- Validate the output with `inspect_music_signal_file`: duration, non-silence, low DC/clipping,
  harmonic structure, and no rotor-like fast repetition.
- Keep dependencies optional; do not move ACE-Step, MLX, Gradio, or codec dependencies into base.
- Record runtime metadata: backend, model id, device, LM backend, precision, duration, seed, and
  quality-smoke metrics.

## Scope

- New backend module, for example `src/abstractmusic/backends/acestep_official.py`.
- CLI/backend factory wiring with a distinct backend kind, for example `acestep-official`.
- Registry entry or backend kind update for `ACE-Step/Ace-Step1.5`.
- Integration test coverage gated by `ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1`.
- Documentation for Apple MLX/MPS and CUDA execution.

## Non-goals

- Do not make the partial custom backend recommended.
- Do not require Gradio or launch a web server for in-process generation.
- Do not make non-commercial or license-unclear models defaults.

## Dependencies and related tasks

- `docs/backlog/completed/040_real_generation_validation_matrix.md`
- `docs/backlog/planned/035_acestep_v15_backend_compatibility_hardening.md`
- `docs/backlog/planned/050_dependency_profiles_and_optional_providers.md`

## Expected outcomes

- AbstractMusic has one validated local open-source music provider.
- The model abstraction remains stable while provider-specific ACE-Step details stay inside the
  backend.
- The custom DiT-only path is either retired or clearly marked experimental.

## Validation

- `python -m pytest -q -m "not integration"`
- `ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1 ABSTRACTMUSIC_REAL_BACKEND=acestep-official
  ABSTRACTMUSIC_REAL_DEVICE=mps ABSTRACTMUSIC_REAL_DURATION_S=10 python -m pytest -q
  tests/integration/test_real_generation.py`
- Manual inspection of the generated WAV metrics and, where possible, human listening review.

## Progress checklist

- [x] Prototype direct upstream handler generation using MLX for the 5Hz LM on Apple Silicon.
- [x] Implement the backend behind an explicit optional extra.
- [x] Add CLI and AbstractCore plugin wiring.
- [x] Run and record a 10-second music-like smoke.
- [x] Update the registry recommendation to prefer the official backend while keeping the custom
  backend non-recommended.

## Progress notes

2026-05-15: Direct upstream prototype succeeded on Apple Silicon using:

- DiT/VAE/text encoder from `ACE-Step/Ace-Step1.5`.
- 5Hz LM from official `ACE-Step/acestep-5Hz-lm-0.6B` because the bundled 1.7B download stalled.
- MLX backend for the LM, MLX DiT, and MLX VAE.
- Runtime no-meta Transformers init-context shim, matching the compatibility issue already handled
  in the custom backend.
- `vector-quantize-pytorch` as a required upstream model-code dependency.

Generated `smoke-artifacts/official-acestep/acestep_official_mlx_10s.wav`. Metrics: 48 kHz stereo,
10.0s, RMS 0.123, peak 0.891, clipped ratio 0, harmonic p75 0.860, dominant envelope 5.0 Hz,
`music_like=True`, `repetitive=False`.

The prototype is not yet an AbstractMusic provider validation because it bypassed the public
backend abstraction.

2026-05-15: Added `src/abstractmusic/backends/acestep_official.py`, CLI engine
`acestep-official`/`official`, REPL LM controls, AbstractCore plugin registration, and the optional
`acestep-official` dependency extra. The initial 1.7B LM download stalled, so the first provider
validation used the official `ACE-Step/acestep-5Hz-lm-0.6B` LM.

2026-05-15: Real AbstractMusic backend validation passed:

- Command: `ABSTRACTMUSIC_RUN_REAL_MODEL_TESTS=1 ABSTRACTMUSIC_REAL_BACKEND=acestep-official
  ABSTRACTMUSIC_REAL_DEVICE=auto ABSTRACTMUSIC_REAL_DURATION_S=10
  ABSTRACTMUSIC_ACESTEP_SOURCE_DIR=/tmp/ACE-Step-1.5-main
  ABSTRACTMUSIC_ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
  ABSTRACTMUSIC_ACESTEP_LM_BACKEND=mlx python -m pytest -q
  tests/integration/test_real_generation.py -s`
- Artifact: `smoke-artifacts/official-backend/acestep-official_smoke.wav`
- Runtime path: MLX 5Hz LM, MLX DiT, MLX VAE.
- Metrics: 48 kHz stereo, 10.0s, peak 0.891, RMS 0.143, clipped ratio 0,
  harmonic p75 0.714, voiced frame ratio 0.970, dominant envelope 5.0 Hz,
  `music_like=True`, `repetitive=False`.

2026-05-15: Retried the 1.7B LM through Hugging Face Hub with Xet disabled, activated the real
`ACE-Step/Ace-Step1.5/acestep-5Hz-lm-1.7B` checkpoint, and changed the official backend default to
`acestep-5Hz-lm-1.7B`.

- Command: `abstractmusic --engine official --device mps --lm-backend mlx --duration 10 --steps 8
  --seed 170017 t2m "relaxing instrumental music with soft piano, warm pads, gentle rhythm, clear
  harmonic progression" --lyrics "[Instrumental]" --out
  smoke-artifacts/user-repl/official-1p7b-10s.wav`
- Artifact: `smoke-artifacts/user-repl/official-1p7b-10s.wav`
- Runtime path: MLX 5Hz LM, MLX DiT, MLX VAE.
- Metrics: 48 kHz stereo, 10.0s, peak 0.891, RMS 0.152, clipped ratio 0,
  harmonic p75 0.913, tonal frame ratio 0.947, voiced frame ratio 1.000,
  dominant envelope 0.501 Hz, `music_like=True`, `repetitive=False`.

2026-05-15: Follow-up listening feedback found the 1.7B short instrumental output highly
repetitive despite passing harmonic/noise checks. The official backend was aligned with upstream
turbo scheduling by defaulting `shift=3.0`, and the CLI/REPL now expose `shift`, `infer_method`,
LM sampling controls, and `audio_cover_strength`. Additional trials showed that the provider still
needs a separate repetition/novelty gate before it can be called production-quality.

2026-05-15: The official API example shape was also run locally on Apple hardware with
`caption="upbeat electronic dance music with heavy bass"`, `bpm=128`, `duration=30`, 0.6B LM, empty
lyrics, and batch generation. It required AbstractMusic's Transformers meta-init compatibility shim
to load on this environment. The run succeeded, but the generated WAVs still showed very high
spectral self-similarity. AbstractMusic was updated to stop forcing `[Instrumental]` when lyrics are
omitted and to pass explicit BPM/key/time-signature metadata.

## Guidance for the implementing agent

Favor the official ACE-Step handler and model planner over reconstructing model internals. Keep the
integration small, explicit, and honest about hardware fallbacks.
