# AbstractMusic Backlog Overview

## Current status

AbstractMusic has a thin public manager, a minimal backend protocol, a generic Diffusers audio
backend, an official ACE-Step v1.5 adapter, an ACE-Step v1.5 custom backend, an ACE-Step Diffusers
adapter, and an AbstractCore capability plugin. The custom ACE-Step smoke is considered a failed
music-quality validation: it produced valid PCM that sounded and measured like fast rotor-like
audio.

The repository has the baseline docs, hygiene files, import-light packaging, a model registry, a
CLI REPL, and a real smoke validation harness now. A direct upstream ACE-Step v1.5 prototype with
the official 0.6B 5Hz LM on MLX generated a 10-second music-like WAV on 2026-05-15, and that path
is now wrapped as `acestep-official`. The 1.7B 5Hz LM has since been downloaded, validated through
the CLI, and made the default because the 0.6B results were too weak for the target quality bar.

## Status counts

- Planned: 9
- Proposed: 1
- Completed: 5
- Deprecated: 0
- Recurrent: 2

## Priority bands

- P0: keep the validated official ACE-Step path reliable and user-tryable through the REPL.
- P0: add a repetition/novelty quality gate before calling ACE-Step output production-quality.
- P1: add a small MusicGen validation backend so ACE-Step quality can be judged against a known
  small text-to-music baseline.
- P1: broaden provider coverage and improve quality metadata with real validation.
- P2: evaluate optional/non-core model families without polluting the base install.

## Next recommended work

1. Complete `planned/065_acestep_repetition_quality_gate.md`.
2. Complete `planned/070_musicgen_small_optional_provider.md`.
3. Complete `planned/045_audio_artifact_screening_and_quality_metadata.md`.
4. Complete `planned/030_acestep_diffusers_xl_provider.md`.
5. Complete `planned/035_acestep_v15_backend_compatibility_hardening.md`.
6. Complete `planned/050_dependency_profiles_and_optional_providers.md`.
7. Complete `planned/055_heartmula_optional_provider.md`.
8. Complete `planned/060_yue_optional_provider.md`.
9. Complete `planned/075_stable_audio_open_small_validation.md`.

## Planned work

| Priority | Item | Outcome |
| --- | --- | --- |
| P1 | `planned/030_acestep_diffusers_xl_provider.md` | Add and validate a dedicated ACE-Step Diffusers provider for the official XL Turbo checkpoint. |
| P1 | `planned/035_acestep_v15_backend_compatibility_hardening.md` | Stabilize or retire the current custom ACE-Step v1.5 backend with tested dependency bounds. |
| P1 | `planned/045_audio_artifact_screening_and_quality_metadata.md` | Strengthen artifact screening and make quality validation metadata first-class. |
| P0 | `planned/065_acestep_repetition_quality_gate.md` | Add a spectral novelty gate so repetitive harmonic loops are not treated as acceptable music. |
| P1 | `planned/070_musicgen_small_optional_provider.md` | Add a small non-commercial MusicGen baseline provider for real quality comparison. |
| P2 | `planned/075_stable_audio_open_small_validation.md` | Validate the gated Stable Audio Open Small short-clip provider through AbstractMusic. |
| P2 | `planned/050_dependency_profiles_and_optional_providers.md` | Move heavy model stacks behind extras and document TinyMozart/Omni2Sound boundaries. |
| P2 | `planned/055_heartmula_optional_provider.md` | Evaluate and implement HeartMuLa as an optional lyrics/tags music provider if dependency and runtime boundaries are acceptable. |
| P2 | `planned/060_yue_optional_provider.md` | Evaluate YuE as an optional multi-stage lyrics-to-music provider without claiming partial token generation is audio. |

## Completed work

| Completed | Item | Outcome |
| --- | --- | --- |
| 2026-05-15 | `completed/000_critical_assessment_and_roadmap.md` | Preserved the assessment and created the implementation roadmap. |
| 2026-05-15 | `completed/010_repo_hygiene_docs_and_packaging.md` | Added repo/docs baseline, license/security/contribution files, ignore rules, and cleaned generated tracked artifacts. |
| 2026-05-15 | `completed/020_music_abstraction_and_capability_registry.md` | Added capability types, request fields, backend capability hooks, packaged model registry, and registry tests. |
| 2026-05-15 | `completed/025_official_acestep_v15_mlx_provider.md` | Added and validated the official ACE-Step backend through the public abstraction using MLX LM/DiT/VAE on Apple Silicon. |
| 2026-05-15 | `completed/040_real_generation_validation_matrix.md` | Added opt-in real generation tests and tightened WAV/music-likeness inspection after a short MPS smoke failed listening review. |

## Proposed work

| Item | Promotion criteria |
| --- | --- |
| `proposed/2026-05-08_music_install_profile_boundary.md` | Promote when the dependency-profile split becomes implementation work. |

## Completion process

When a planned item is complete:

1. Add a completion report to the item.
2. Set `Status: Completed` and `Completed: <YYYY-MM-DD>`.
3. Move it from `planned/` to `completed/`.
4. Update this overview, including counts and any changed priorities.
5. Run recurrent tasks whose conditions apply.

## Planning notes

- ACE-Step v1.5 should use `acestep-official`; the older custom backend remains non-recommended
  because it produced rotor-like audio.
- `acestep-official` passed 10-second harmonic music smokes through AbstractMusic on 2026-05-15
  using both official 0.6B and 1.7B LMs with MLX LM, MLX DiT, and MLX VAE. The 1.7B LM is now the
  default.
- On Apple hardware, prefer an official MLX path when a provider actually supports it; otherwise
  use PyTorch MPS. CPU should be a clearly marked fallback, not the default accelerated path.
- No additional model should be marked recommended until it passes a real generation smoke test
  through AbstractMusic.
- The official ACE-Step XL Turbo Diffusers checkpoint is the cleanest near-term improvement path
  because it uses a standard Diffusers pipeline layout and `AceStepPipeline`.
- `facebook/musicgen-small` is a useful small validation provider because it has a simple
  Transformers path and 300M weights, but it is CC BY-NC 4.0 and must not become the commercial
  default.
- `stabilityai/stable-audio-open-small` is a 341M gated short text-to-audio model worth tracking
  for sound effects/loops and Apple/Arm-friendly experiments, but it is not currently a strong
  default music model candidate.
- TinyMozart is small and local but is unconditional MIDI piano, not general prompt-to-music, and
  has no declared license in the Hugging Face metadata reviewed on 2026-05-15.
- Omni2Sound is useful research for multimodal audio/foley, but it is CC BY-NC 4.0, CUDA/script
  oriented, very large, and not suitable for a default commercial-capable music provider.
- HeartMuLa can generate lyrics/tags conditioned music according to its Apache-2.0 Hugging Face
  model card and official `heartlib` deployment docs. It should be evaluated as an optional
  provider because it requires HeartCodec and a heavy runtime stack.
- YuE Stage-1 is music-relevant and Apache-2.0, but it is not an end-to-end audio generator by
  itself. A YuE provider needs the Stage-2 and codec/upsampler path before it can satisfy
  AbstractMusic's generation goal.
