# AbstractMusic Backlog Overview

## Current status

AbstractMusic has a thin public manager, a minimal backend protocol, stdlib-only ACE Music and
ElevenLabs Music remote backends, a generic Diffusers audio backend, a standalone ACE-Step v1.5
backend, an ACE-Step Diffusers adapter, an internal Stable Audio 3 Small Music spike, and an
AbstractCore capability plugin. The ACE-Step paths must not call a local ACE-Step source checkout
or external ACE-Step package, and the Stable Audio 3 path must not import or wrap the upstream
Stable Audio runtime package.

The repository has the baseline docs, hygiene files, import-light packaging, a model registry, a
CLI REPL, and a real smoke validation harness. Previous external-runtime experiments are no longer
supported provider paths; accepted reference WAVs remain comparison artifacts only.

## Status counts

- Planned: 9
- Proposed: 2
- Completed: 5
- Deprecated: 2
- Recurrent: 2

## Priority bands

- P0: make the standalone ACE-Step backend produce usable music through internal package code.
- P0: add a repetition/novelty quality gate before calling ACE-Step output production-quality.
- P1: add a small MusicGen validation backend so ACE-Step quality can be judged against a known
  small text-to-music baseline.
- P1: broaden Stable Audio 3.0 Small validation beyond the focused 30-second and 120-second
  smokes before considering Medium or recommendation status.
- P1: broaden provider coverage and improve quality metadata with real validation.
- P2: evaluate optional/non-core model families without polluting the base install.

## Next recommended work

1. Complete `planned/065_acestep_repetition_quality_gate.md`.
2. Complete the broader validation in `planned/0083_stable_audio_3_local_provider_spike.md`.
3. Complete `planned/070_musicgen_small_optional_provider.md`.
4. Complete `planned/045_audio_artifact_screening_and_quality_metadata.md`.
5. Complete `planned/030_acestep_diffusers_xl_provider.md`.
6. Complete `planned/035_acestep_v15_backend_compatibility_hardening.md`.
7. Complete `planned/055_heartmula_optional_provider.md`.
8. Complete `planned/060_yue_optional_provider.md`.
9. Complete `planned/075_stable_audio_open_small_validation.md`.

## Planned work

| Priority | Item | Outcome |
| --- | --- | --- |
| P1 | `planned/0083_stable_audio_3_local_provider_spike.md` | Broaden Stable Audio 3.0 Small validation after focused 30-second and 120-second implementation smokes, then decide whether Medium should follow. |
| P1 | `planned/030_acestep_diffusers_xl_provider.md` | Add and validate a dedicated ACE-Step Diffusers provider for the official XL Turbo checkpoint. |
| P1 | `planned/035_acestep_v15_backend_compatibility_hardening.md` | Stabilize or retire the current custom ACE-Step v1.5 backend with tested dependency bounds. |
| P1 | `planned/045_audio_artifact_screening_and_quality_metadata.md` | Strengthen artifact screening and make quality validation metadata first-class. |
| P0 | `planned/065_acestep_repetition_quality_gate.md` | Add a spectral novelty gate so repetitive harmonic loops are not treated as acceptable music. |
| P1 | `planned/070_musicgen_small_optional_provider.md` | Add a small non-commercial MusicGen baseline provider for real quality comparison. |
| P2 | `planned/075_stable_audio_open_small_validation.md` | Validate the gated Stable Audio Open Small short-clip provider through AbstractMusic. |
| P2 | `planned/055_heartmula_optional_provider.md` | Evaluate and implement HeartMuLa as an optional lyrics/tags music provider if dependency and runtime boundaries are acceptable. |
| P2 | `planned/060_yue_optional_provider.md` | Evaluate YuE as an optional multi-stage lyrics-to-music provider without claiming partial token generation is audio. |

## Completed work

| Completed | Item | Outcome |
| --- | --- | --- |
| 2026-05-15 | `completed/000_critical_assessment_and_roadmap.md` | Preserved the assessment and created the implementation roadmap. |
| 2026-05-15 | `completed/010_repo_hygiene_docs_and_packaging.md` | Added repo/docs baseline, license/security/contribution files, ignore rules, and cleaned generated tracked artifacts. |
| 2026-05-15 | `completed/020_music_abstraction_and_capability_registry.md` | Added capability types, request fields, backend capability hooks, packaged model registry, and registry tests. |
| 2026-05-15 | `completed/040_real_generation_validation_matrix.md` | Added opt-in real generation tests and tightened WAV/music-likeness inspection after a short MPS smoke failed listening review. |
| 2026-05-21 | `completed/050_dependency_profiles_and_optional_providers.md` | Added the lightweight ACE Music remote backend, kept base dependencies empty, expanded local platform extras, and documented optional provider boundaries. |

## Proposed work

| Item | Promotion criteria |
| --- | --- |
| `proposed/0080_text_planning_provider_contract_for_music.md` | Promote when advanced music quality requires LLM-generated captions/lyrics/metadata and a no-AbstractCore-dependency planner boundary is clear. |
| `proposed/0082_local_engine_priority_after_remote_baseline.md` | Promote when choosing the next local/open-weight engine spike after the two remote endpoint baseline. |

## Deprecated work

| Deprecated | Item | Reason |
| --- | --- | --- |
| 2026-05-20 | `deprecated/0025_external_acestep_runtime_wrapper.md` | Removed the out-of-package ACE-Step runtime wrapper path; standalone package code is required. |
| 2026-05-21 | `deprecated/0081_music_install_profile_boundary.md` | Superseded by completed dependency-profile implementation. |

## Completion process

When a planned item is complete:

1. Add a completion report to the item.
2. Set `Status: Completed` and `Completed: <YYYY-MM-DD>`.
3. Move it from `planned/` to `completed/`.
4. Update this overview, including counts and any changed priorities.
5. Run recurrent tasks whose conditions apply.

## Planning notes

- ACE-Step v1.5 must use the standalone `acestep` / `acestep-v15` package backend.
- Previous external-runtime smoke artifacts remain useful as references, but they do not prove the
  standalone package path works.
- On Apple hardware, prefer PyTorch MPS for the standalone ACE-Step backend with clear CPU
  fallbacks for known unstable text-encoder/decode steps.
- No additional model should be marked recommended until it passes a real generation smoke test
  through AbstractMusic.
- The ACE-Step XL Turbo Diffusers checkpoint is the cleanest near-term improvement path
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
- With ACE Music and ElevenLabs Music in the base package, remote coverage is sufficient for now.
  Future provider work should prioritize local/open-weight generation unless a remote endpoint
  proves a reusable abstraction needed by local engines.
- MiniMax documents a `music-2.6-free` API model, but its pricing and access pages still frame
  music usage around token plans, pay-as-you-go balance, and limited-free quotas. Do not add it as
  a third remote backend unless a user explicitly wants that provider and accepts the access
  ambiguity.
- Stable Audio 3.0 Small now has an internal package-owned text-to-music path that passed focused
  30-second and 120-second validation runs at 16 steps. It is still a spike, not a recommended
  default, until broader prompt/seed and GPU validation are recorded.
