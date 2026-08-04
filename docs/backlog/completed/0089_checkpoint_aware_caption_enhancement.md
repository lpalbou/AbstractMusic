# Proposed: Checkpoint-aware caption enhancement for guided ACE-Step variants

## Metadata
- Created: 2026-08-04
- Status: Completed
- Completed: 2026-08-04

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context

Controlled A/B generations (2026-08-04) established that long template-expanded captions
reliably degrade the guided ACE-Step XL checkpoints (`acestep-v15-xl-sft-diffusers`,
`acestep-v15-xl-base-diffusers`, guidance_scale 7.0) — sometimes into unpitched noise-sweep
collapse confirmed by listening — while the same seeds with raw short prompts generate music.
The profile's bpm/keyscale/timesignature hints were isolated separately and are harmless. The
guidance-distilled turbo checkpoint tolerates the expanded captions and, on one arcade control
pair, produced better-gated output *with* the expansion than without it.

The deterministic planner no longer forces expansion for style profiles, which fixes the default
30-second path. Two opted-in paths still enable the same caption bundle by design:

- long-form structure maps (`--structure-prompt`, default-on, activating at >= 45 s);
- `--auto-lyrics`.

On guided XL checkpoints those paths carry the same degradation risk. Nothing at planning time
knows which checkpoint will consume the caption, even though `MusicPlanningRequest` already
carries `backend` and `model_id`, and the packaged registry knows which variants are
guidance-distilled.

## Problem or opportunity

The caption policy that helps one checkpoint family harms another, and the planner currently
applies one policy to all. Long-form (>= 45 s) generation on XL sft/base with default flags is
the concrete exposure.

## Proposed direction

- Record a caption-tolerance hint per registry entry (for example
  `caption_style: distilled-long | guided-short`), sourced from the validation record.
- Let the deterministic planner consult the hint through the `model_id` it already receives:
  keep structure maps for long-form but render them compactly (section list only, no template
  prose) for guided variants.
- Validate with the existing gate set (`evaluate_music_quality_gates`) on a >= 60 s matrix per
  variant before changing any default.

## Non-goals

- No LLM planner dependencies; the deterministic planner stays dependency-free.
- No per-model prompt engineering beyond the registry hint; provider-specific logic stays out
  of the planning layer.

## Completion report (2026-08-04)

Implemented as a `caption_sensitive` boolean on registry entries (set for
`acestep-v15-xl-sft-diffusers` and `acestep-v15-xl-base-diffusers`) consulted by the
deterministic planner through the `model_id` it already receives. For caption-sensitive
checkpoints, features that need a caption (long-form structure maps, auto-lyrics) render it
compactly — the raw prompt plus the section map, none of the template prose. An explicit
`--enhance-prompt` still applies the full expansion, with a
`caption_expansion_on_caption_sensitive_model` provenance warning; the compact path records
`compact_caption_for_caption_sensitive_model`. The CLI resolves the backend's default model id
so sensitivity is seen even without `--model-id`.

Validation (XL sft, 60 s, seed 123, MPS bfloat16, arcade prompt): full bundle, compact, and raw
all passed the complete automated gate set; the full-bundle control subsequently FAILED a
maintainer listening review ("much too fast until maybe 20s"; tracker localizes a 2:1 onset-rate plateau over ~0-12 s), which no automated gate
measures — see backlog 0090 for the tempo-stability gate gap. Notably the 30-second collapse did not reproduce at 60 s in
the full-bundle control — the >=45 s bundle includes a concrete section map rather than pure
instruction prose — so the compact default is a conservative measure that removes the
proven-risky prose while keeping the section map, not a fix for a demonstrated 60 s failure.
Unit tests pin the compact rendering, the explicit-enhance override, and the auto-lyrics path.
The proposal's per-variant 60 s matrix bar is only partially met: XL sft has the three-way 60 s
control; XL base has no 60 s runs yet. Injected LLM planners own their caption content; a long
planner caption aimed at a sensitive checkpoint is recorded in provenance
(`long_planner_caption_on_caption_sensitive_model`) rather than rewritten.
