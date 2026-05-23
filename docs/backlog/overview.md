# AbstractMusic Backlog Overview

## Current status

AbstractMusic has a thin public manager, a minimal backend protocol, stdlib-only ACE Music and
ElevenLabs Music remote backends, a generic Diffusers audio backend, a single public ACE-Step
backend built on `AceStepPipeline`, an internal Stable Audio 3 Small Music spike, and an
AbstractCore capability plugin. The ACE-Step path must not call a local ACE-Step source checkout or
external ACE-Step package, and the Stable Audio 3 path must not import or wrap the upstream Stable
Audio runtime package.

The repository has the baseline docs, hygiene files, import-light packaging, a model registry, a
CLI REPL, and a real smoke validation harness. Previous external-runtime experiments are no longer
supported provider paths; accepted reference WAVs remain comparison artifacts only.

## Status counts

- Planned: 2
- Proposed: 3
- Completed: 12
- Deprecated: 5
- Recurrent: 2

## Priority bands

- P1: keep provider routing, defaults, and duration handling truthful and robust.
- P1: broaden local-engine validation (Stable Audio 3 and ACE-Step Diffusers) before changing any
  recommendation/default status.
- P2: evaluate optional/non-core model families without polluting the base install.

## Next recommended work

1. Complete `planned/055_heartmula_optional_provider.md`.
2. Complete `planned/060_yue_optional_provider.md`.

## Planned work

| Priority | Item | Outcome |
| --- | --- | --- |
| P2 | `planned/055_heartmula_optional_provider.md` | Evaluate and implement HeartMuLa as an optional lyrics/tags music provider if dependency and runtime boundaries are acceptable. |
| P2 | `planned/060_yue_optional_provider.md` | Evaluate YuE as an optional multi-stage lyrics-to-music provider without claiming partial token generation is audio. |

## Completed work

| Completed | Item | Outcome |
| --- | --- | --- |
| 2026-05-15 | `completed/000_critical_assessment_and_roadmap.md` | Preserved the assessment and created the implementation roadmap. |
| 2026-05-15 | `completed/010_repo_hygiene_docs_and_packaging.md` | Added repo/docs baseline, license/security/contribution files, ignore rules, and cleaned generated tracked artifacts. |
| 2026-05-15 | `completed/020_music_abstraction_and_capability_registry.md` | Added capability types, request fields, backend capability hooks, packaged model registry, and registry tests. |
| 2026-05-15 | `completed/040_real_generation_validation_matrix.md` | Added opt-in real generation tests and tightened WAV/music-likeness inspection after a short MPS smoke failed listening review. |
| 2026-05-21 | `completed/030_acestep_diffusers_xl_provider.md` | Added and validated the package-owned ACE-Step XL Turbo path that now backs the public `acestep` backend. |
| 2026-05-21 | `completed/045_audio_artifact_screening_and_quality_metadata.md` | Strengthened artifact screening and made validation state more explicit through smoke metrics, tests, and registry status. |
| 2026-05-21 | `completed/050_dependency_profiles_and_optional_providers.md` | Added the lightweight ACE Music remote backend, kept base dependencies empty, expanded local platform extras, and documented optional provider boundaries. |
| 2026-05-21 | `completed/065_acestep_repetition_quality_gate.md` | Added repetition/novelty-oriented artifact screening and wired it into ACE-Step quality reporting. |
| 2026-05-21 | `completed/0083_stable_audio_3_local_provider_spike.md` | Implemented and validated Stable Audio 3 Small Music as a package-owned local backend (Small validated; Medium deferred). |
| 2026-05-21 | `completed/0084_music_capability_residency_contract.md` | Exposed Core-friendly load/list/unload residency for local music engines without confusing remote discovery with local loaded state. |
| 2026-05-21 | `completed/0085_truthful_stable_audio_capability_registration_and_music_routing.md` | Registered Stable Audio Open Small as a real capability backend and aligned discovery/catalog routing truth. |
| 2026-05-21 | `completed/0086_repl_ux_and_default_model_routing_hardening.md` | Made the REPL discoverable and robust (engine/model routing, aligned model listing, downloads toggle, and data-driven defaults). |

## Proposed work

| Item | Promotion criteria |
| --- | --- |
| `proposed/0080_text_planning_provider_contract_for_music.md` | Promote when advanced music quality requires LLM-generated captions/lyrics/metadata and a no-AbstractCore-dependency planner boundary is clear. |
| `proposed/0082_local_engine_priority_after_remote_baseline.md` | Promote when choosing the next local/open-weight engine spike after the two remote endpoint baseline. |
| `proposed/0087_truthful_music_provider_runtime_availability.md` | Promote before clients rely on provider lists for selectable music backends; provider availability must mean runnable in the connected deployment. |

## Deprecated work

| Deprecated | Item | Reason |
| --- | --- | --- |
| 2026-05-20 | `deprecated/0025_external_acestep_runtime_wrapper.md` | Removed the out-of-package ACE-Step runtime wrapper path; standalone package code is required. |
| 2026-05-21 | `deprecated/035_acestep_v15_backend_compatibility_hardening.md` | Superseded by the single-backend `acestep` contract; the old standalone `acestep-v15` path is no longer part of the supported surface. |
| 2026-05-21 | `deprecated/0081_music_install_profile_boundary.md` | Superseded by completed dependency-profile implementation. |
| 2026-05-21 | `deprecated/0075_stable_audio_open_small_validation.md` | Open Small validation is no longer tracked as separate planned work; keep it as an optional legacy backend and focus validation on Stable Audio 3. |
| 2026-05-21 | `deprecated/070_musicgen_small_optional_provider.md` | MusicGen Small remains an optional non-commercial backend, but we are not pursuing further validation/recommendation work as part of the core roadmap. |

## Completion process

When a planned item is complete:

1. Add a completion report to the item.
2. Set `Status: Completed` and `Completed: <YYYY-MM-DD>`.
3. Move it from `planned/` to `completed/`.
4. Update this overview, including counts and any changed priorities.
5. Run recurrent tasks whose conditions apply.

## Planning notes

- ACE-Step discovery/UI must stay on the single public `acestep` backend contract.
- Previous external-runtime smoke artifacts remain useful as references, but they do not define the
  supported package path.
- On Apple hardware, prefer PyTorch MPS for the `acestep` backend with clear CPU fallbacks for
  known unstable decode steps.
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
- Stable Audio Open Small is registered and routed truthfully through the AbstractCore plugin as
  `abstractmusic:stable-audio`, but it remains a gated short-clip legacy backend and is not marked
  recommended.
